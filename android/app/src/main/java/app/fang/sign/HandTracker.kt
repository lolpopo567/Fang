package app.fang.sign

import android.content.Context
import android.graphics.Bitmap
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult
import java.io.Closeable

/** Pairs of landmark indices joined in the overlay drawing (MediaPipe's HAND_CONNECTIONS). */
val HAND_CONNECTIONS = listOf(
    0 to 1, 1 to 2, 2 to 3, 3 to 4, 0 to 5, 5 to 6, 6 to 7, 7 to 8, 5 to 9, 9 to 10, 10 to 11, 11 to 12,
    9 to 13, 13 to 14, 14 to 15, 15 to 16, 13 to 17, 0 to 17, 17 to 18, 18 to 19, 19 to 20,
)

/**
 * MediaPipe Hand Landmarker on the device, with the same model file and settings as the web app (2 hands,
 * video mode). Tries the GPU first and falls back to the CPU, as the web app does.
 */
class HandTracker(context: Context) : Closeable {
    private val landmarker: HandLandmarker
    val delegate: String
    private var lastTimestamp = -1L

    init {
        var made: HandLandmarker? = null
        var used = "GPU"
        try {
            made = create(context, Delegate.GPU)
        } catch (_: Throwable) {
            used = "CPU"
        }
        landmarker = made ?: create(context, Delegate.CPU)
        delegate = used
    }

    private fun create(context: Context, delegate: Delegate): HandLandmarker {
        val base = BaseOptions.builder().setModelAssetPath("hand_landmarker.task").setDelegate(delegate).build()
        val options = HandLandmarker.HandLandmarkerOptions.builder()
            .setBaseOptions(base)
            .setRunningMode(RunningMode.VIDEO)
            .setNumHands(2)
            .build()
        return HandLandmarker.createFromOptions(context, options)
    }

    /**
     * [bitmap] must be upright and NOT mirrored, even for the front camera: the web app feeds MediaPipe the
     * raw camera frame, and the models learned its Left/Right labels from that.
     */
    fun detect(bitmap: Bitmap, timestampMs: Long): HandFrame {
        // Video mode needs strictly increasing timestamps.
        val ts = maxOf(timestampMs, lastTimestamp + 1)
        lastTimestamp = ts
        val result = landmarker.detectForVideo(BitmapImageBuilder(bitmap).build(), ts)
        return toHandFrame(result, ts.toDouble(), bitmap.width.toDouble() / bitmap.height)
    }

    override fun close() = landmarker.close()
}

fun toHandFrame(result: HandLandmarkerResult, t: Double, aspect: Double): HandFrame {
    val hands = result.landmarks().mapIndexed { i, lms ->
        val cat = result.handedness().getOrNull(i)?.firstOrNull()
        Hand(
            handedness = cat?.categoryName() ?: "Unknown",
            score = cat?.score()?.toDouble() ?: 0.0,
            landmarks = lms.map { doubleArrayOf(it.x().toDouble(), it.y().toDouble(), it.z().toDouble()) },
        )
    }
    return HandFrame(t, aspect, hands)
}
