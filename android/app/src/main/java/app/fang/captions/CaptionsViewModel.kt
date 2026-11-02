package app.fang.captions

import android.app.Application
import android.content.Intent
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.os.VibrationEffect
import android.os.Vibrator
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.AndroidViewModel
import app.fang.settings.Prefs

/**
 * Live captions with Android's SpeechRecognizer. Like the web app, it restarts itself after each pause until
 * the user stops it. With "recognise on this phone" on, it asks for the offline language pack, so audio
 * stays on the phone. Status and errors are i18n keys; the screen turns them into text.
 */
class CaptionsViewModel(app: Application) : AndroidViewModel(app) {
    private val prefs = Prefs.get(app)
    private val main = Handler(Looper.getMainLooper())

    val lines = mutableStateListOf<String>()
    var interim by mutableStateOf("")
        private set
    var listening by mutableStateOf(false)
        private set
    /** An i18n key, or "" for none. */
    var status by mutableStateOf("")
    var statusDetail by mutableStateOf("")
        private set

    val supported = SpeechRecognizer.isRecognitionAvailable(app)
    private var recognizer: SpeechRecognizer? = null
    private var backoffMs = 250L

    fun toggle() = if (listening) stop() else start()

    fun start() {
        if (!supported) return
        listening = true
        backoffMs = 250L
        setStatus("captions.listening")
        listen()
    }

    fun stop() {
        listening = false
        main.removeCallbacksAndMessages(null)
        recognizer?.destroy()
        recognizer = null
        interim = ""
        setStatus("captions.stopped")
    }

    fun clear() {
        lines.clear()
        interim = ""
    }

    fun transcript() = lines.joinToString("\n")

    /** The spoken language or the offline setting changed: restart so it takes effect. */
    fun settingsChanged() {
        if (listening) {
            recognizer?.cancel()
            restartSoon(0)
        }
    }

    private fun setStatus(key: String, detail: String = "") {
        status = key
        statusDetail = detail
    }

    private fun listen() {
        if (!listening) return
        val rec = recognizer ?: SpeechRecognizer.createSpeechRecognizer(getApplication()).also {
            it.setRecognitionListener(listener)
            recognizer = it
        }
        val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
            putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
            putExtra(RecognizerIntent.EXTRA_LANGUAGE, prefs.captionLang)
            putExtra(RecognizerIntent.EXTRA_PARTIAL_RESULTS, true)
            putExtra(RecognizerIntent.EXTRA_PREFER_OFFLINE, prefs.captionOffline)
            putExtra(RecognizerIntent.EXTRA_MAX_RESULTS, 1)
        }
        rec.startListening(intent)
    }

    private fun restartSoon(delayMs: Long = 250) {
        main.removeCallbacksAndMessages(null)
        main.postDelayed({ listen() }, delayMs)
    }

    private val listener = object : RecognitionListener {
        override fun onReadyForSpeech(params: Bundle?) {
            backoffMs = 250L
            if (status != "captions.listening") setStatus("captions.listening")
        }

        override fun onPartialResults(partialResults: Bundle?) {
            interim = partialResults?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)?.firstOrNull()?.trim().orEmpty()
        }

        override fun onResults(results: Bundle?) {
            val text = results?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)?.firstOrNull()?.trim().orEmpty()
            interim = ""
            if (text.isNotEmpty()) {
                lines.add(text)
                buzz()
            }
            // Android ends a session after each pause; keep captions running until the user stops them.
            if (listening) restartSoon()
        }

        override fun onError(error: Int) {
            interim = ""
            if (!listening) return
            when (error) {
                SpeechRecognizer.ERROR_INSUFFICIENT_PERMISSIONS -> fail("captions.err.not-allowed")
                SpeechRecognizer.ERROR_AUDIO -> fail("captions.err.audio-capture")
                SpeechRecognizer.ERROR_LANGUAGE_NOT_SUPPORTED, SpeechRecognizer.ERROR_LANGUAGE_UNAVAILABLE ->
                    fail(if (prefs.captionOffline) "captions.err.offline" else "captions.err.generic", "language")
                SpeechRecognizer.ERROR_NETWORK, SpeechRecognizer.ERROR_NETWORK_TIMEOUT, SpeechRecognizer.ERROR_SERVER -> {
                    // No connection: say so and keep retrying, slower each time.
                    setStatus(if (prefs.captionOffline) "captions.err.offline" else "captions.err.network")
                    backoffMs = (backoffMs * 2).coerceAtMost(8000)
                    restartSoon(backoffMs)
                }
                // Silence, nothing recognised, or the service was busy: just listen again.
                else -> restartSoon()
            }
        }

        override fun onBeginningOfSpeech() {}
        override fun onRmsChanged(rmsdB: Float) {}
        override fun onBufferReceived(buffer: ByteArray?) {}
        override fun onEndOfSpeech() {}
        override fun onEvent(eventType: Int, params: Bundle?) {}
    }

    private fun fail(key: String, detail: String = "") {
        stop()
        setStatus(key, detail)
    }

    private fun buzz() {
        val v = getApplication<Application>().getSystemService(Vibrator::class.java) ?: return
        if (v.hasVibrator()) v.vibrate(VibrationEffect.createOneShot(30, VibrationEffect.DEFAULT_AMPLITUDE))
    }

    override fun onCleared() {
        stop()
    }
}
