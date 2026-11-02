package app.fang.sign

import android.app.Application
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import androidx.camera.core.ImageProxy
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import app.fang.settings.Prefs
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

/** One on-device model per sign language, shared with the web app (assets come from ../models). */
val SIGN_MODELS = mapOf("tsl" to "tsl-model.json", "ase" to "asl-model.json")

// Training data marks "no sign" (resting, transitions) with these labels; they are never shown as words.
val NO_SIGN_LABELS = setOf("null_act", NO_HANDS)
private const val PREDICT_EVERY_MS = 500L
private const val RECORD_MS = 2000L

/**
 * The sign camera pipeline: CameraX frames -> MediaPipe hand landmarks -> HandLabeler -> a time window ->
 * the recognizer (on-device model, online API, or hybrid), every 500 ms, as in the web app's sign.js.
 * Camera frames are analysed on one background thread; UI state is only written on the main thread.
 */
class SignViewModel(app: Application) : AndroidViewModel(app) {
    private val prefs = Prefs.get(app)
    private val main = Handler(Looper.getMainLooper())
    val executor: ExecutorService = Executors.newSingleThreadExecutor()

    // Only touched on the analysis thread.
    private var tracker: HandTracker? = null
    private val window = FrameWindow(30)
    private val labeler = HandLabeler()
    private var fpsCount = 0
    private var fpsStart = 0L
    private var lastPredict = 0L
    @Volatile private var predicting = false
    @Volatile private var recording: MutableList<HandFrame>? = null

    private val offline = OfflineRecognizer()
    private val online = OnlineRecognizer(prefs.signEndpoint)
    private val hub = RecognizerHub(offline, online).also { it.mode = RecognizerMode.of(prefs.signMode) }

    var running by mutableStateOf(false)
        private set
    var front by mutableStateOf(true)
        private set
    /** i18n key for the pipeline status, plus extra text (an error message). */
    var pipeline by mutableStateOf("sign.stopped" to "")
        private set
    var hands by mutableStateOf<Int?>(null)
        private set
    var fps by mutableStateOf<Int?>(null)
        private set
    var latencyMs by mutableStateOf<Long?>(null)
        private set
    var overlay by mutableStateOf<HandFrame?>(null)
        private set
    var prediction by mutableStateOf<Prediction?>(null)
        private set

    val recorder = SampleRecorder(app)
    var recStatus by mutableStateOf("" to "")
    var sampleCount by mutableIntStateOf(recorder.count)
        private set

    private var modelRequest = 0

    init {
        loadModel(prefs.signLang)
    }

    fun loadModel(lang: String) {
        val req = ++modelRequest
        offline.model = null
        prediction = null
        executor.execute { window.clear() }
        viewModelScope.launch {
            val model = withContext(Dispatchers.IO) {
                runCatching {
                    getApplication<Application>().assets.open(SIGN_MODELS[lang] ?: return@runCatching null)
                        .bufferedReader().use { SignModel.parse(it.readText()) }
                }.getOrNull()
            } ?: return@launch
            if (req != modelRequest) return@launch
            offline.model = model
            executor.execute {
                window.size = model.window
                // Models with an fps get a fixed time span (resampled in predict); older ones a fixed frame count.
                window.durationMs = model.fps?.let { 1000.0 * model.window / it } ?: 0.0
                window.clear()
            }
        }
    }

    fun settingsChanged() {
        hub.mode = RecognizerMode.of(prefs.signMode)
        online.endpoint = prefs.signEndpoint
    }

    fun toggle() = if (running) stop() else start()

    fun start() {
        running = true
        pipeline = "sign.loading" to ""
        // MediaPipe is created on the analysis thread and used only there.
        executor.execute {
            try {
                if (tracker == null) tracker = HandTracker(getApplication())
                fpsStart = SystemClock.uptimeMillis()
                fpsCount = 0
                main.post { if (running) pipeline = "sign.running" to "" }
            } catch (e: Throwable) {
                main.post { failed(e) }
            }
        }
    }

    fun stop() {
        running = false
        pipeline = "sign.stopped" to ""
        hands = null
        fps = null
        latencyMs = null
        overlay = null
        executor.execute {
            window.clear()
            labeler.reset()
        }
    }

    fun failed(e: Throwable) {
        stop()
        pipeline = "sign.cameraError" to (e.message ?: e.javaClass.simpleName)
    }

    fun flip() {
        front = !front
    }

    /** CameraX analyzer, on [executor]. */
    fun analyze(proxy: ImageProxy) {
        try {
            val tr = tracker ?: return
            if (!running) return
            val upright = upright(proxy)
            val t0 = SystemClock.uptimeMillis()
            val frame = labeler.push(tr.detect(upright, t0))
            val now = SystemClock.uptimeMillis()
            window.push(frame)
            recording?.add(frame)

            fpsCount++
            val newFps = if (now - fpsStart >= 1000) ((fpsCount * 1000) / (now - fpsStart)).toInt().also {
                fpsCount = 0
                fpsStart = now
            } else null
            main.post {
                if (!running) return@post
                overlay = frame
                hands = frame.hands.size
                latencyMs = now - t0
                if (newFps != null) fps = newFps
            }

            if (!predicting && now - lastPredict > PREDICT_EVERY_MS && window.full) {
                lastPredict = now
                predicting = true
                val frames = window.frames.toList()
                viewModelScope.launch {
                    try {
                        val p = hub.predict(frames)
                        if (running) prediction = p
                    } finally {
                        predicting = false
                    }
                }
            }
        } catch (e: Throwable) {
            main.post { failed(e) }
        } finally {
            proxy.close()
        }
    }

    /**
     * The frame turned upright but NOT mirrored, even from the front camera: the web app gives MediaPipe the
     * raw camera frame, and the sign models learned their Left/Right hand slots from that.
     */
    private fun upright(proxy: ImageProxy): Bitmap {
        val bitmap = proxy.toBitmap()
        val rotation = proxy.imageInfo.rotationDegrees
        if (rotation == 0) return bitmap
        val m = Matrix().apply { postRotate(rotation.toFloat()) }
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, m, true)
    }

    // Dataset sample recorder: landmarks only, no video.
    fun record(label: String) {
        if (!running) return run { recStatus = "rec.needCamera" to "" }
        if (label.isBlank()) return run { recStatus = "rec.needLabel" to "" }
        if (recording != null) return
        val frames = java.util.Collections.synchronizedList(mutableListOf<HandFrame>())
        recording = frames
        recStatus = "rec.recording" to ""
        main.postDelayed({
            recording = null
            val copy = synchronized(frames) { frames.toList() }
            if (copy.none { it.hands.isNotEmpty() }) {
                recStatus = "rec.noHands" to ""
                return@postDelayed
            }
            sampleCount = recorder.add(label.trim(), prefs.signer.trim(), copy)
            recStatus = "rec.count" to sampleCount.toString()
        }, RECORD_MS)
    }

    fun clearSamples() {
        recorder.clear()
        sampleCount = 0
        recStatus = "rec.count" to "0"
    }

    override fun onCleared() {
        main.removeCallbacksAndMessages(null)
        executor.execute {
            tracker?.close()
            tracker = null
        }
        executor.shutdown()
    }
}
