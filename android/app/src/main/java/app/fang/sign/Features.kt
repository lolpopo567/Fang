// Kotlin port of the web app's js/sign/features.js. The numbers must match it exactly, because the
// sign models were trained on features computed that way (checked by ParityTest against the JS output).
package app.fang.sign

import kotlin.math.hypot
import kotlin.math.ln
import kotlin.math.sqrt

const val NUM_LANDMARKS = 21
private const val WRIST = 0
private const val MIDDLE_MCP = 9
private val SLOT_NAMES = listOf("Left", "Right")

/** One tracked hand: MediaPipe's label ("Left"/"Right"), its score, and 21 [x, y, z] image-normalized points. */
class Hand(val handedness: String, val score: Double, val landmarks: List<DoubleArray>) {
    fun withHandedness(label: String) = Hand(label, score, landmarks)
}

/** Landmarks from one camera frame. [t] is in ms; [aspect] is frame width / height. */
class HandFrame(val t: Double, val aspect: Double, val hands: List<Hand>)

private fun hypot3(x: Double, y: Double, z: Double) = sqrt(x * x + y * y + z * z)

// JS `frame.aspect || 1`
private fun aspectOf(frame: HandFrame) = if (frame.aspect == 0.0 || frame.aspect.isNaN()) 1.0 else frame.aspect

/**
 * Translate to the wrist and scale by wrist-to-middle-knuckle distance, after rescaling x and z by the
 * aspect ratio so a portrait phone and a landscape webcam give the same hand shape.
 */
fun normalizeHand(landmarks: List<DoubleArray>, aspect: Double = 1.0): List<DoubleArray>? {
    if (landmarks.size != NUM_LANDMARKS) return null
    val p = landmarks.map { doubleArrayOf(it[0] * aspect, it[1], it[2] * aspect) }
    val w = p[WRIST]
    val m = p[MIDDLE_MCP]
    val scale = hypot3(m[0] - w[0], m[1] - w[1], m[2] - w[2]).takeIf { it != 0.0 } ?: 1.0
    return p.map { doubleArrayOf((it[0] - w[0]) / scale, (it[1] - w[1]) / scale, (it[2] - w[2]) / scale) }
}

/**
 * Fixed-length vector for one frame: left hand 63 values, right hand 63, left present, right present (128).
 * With [handPosition], 6 more: wrist x, y from the frame centre and log(hand size / 0.1), per hand (134).
 */
fun frameVector(frame: HandFrame, handPosition: Boolean = false): DoubleArray {
    val n = NUM_LANDMARKS * 3
    val out = DoubleArray(2 * n + 2 + if (handPosition) 6 else 0)
    val aspect = aspectOf(frame)
    for (hand in frame.hands) {
        val slot = when (hand.handedness) {
            "Left" -> 0
            "Right" -> 1
            else -> continue
        }
        val norm = normalizeHand(hand.landmarks, aspect) ?: continue
        val base = slot * n
        norm.forEachIndexed { i, (x, y, z) ->
            out[base + i * 3] = x
            out[base + i * 3 + 1] = y
            out[base + i * 3 + 2] = z
        }
        out[2 * n + slot] = 1.0
        if (handPosition) {
            val w = hand.landmarks[WRIST]
            val m = hand.landmarks[MIDDLE_MCP]
            val size = hypot3((m[0] - w[0]) * aspect, m[1] - w[1], (m[2] - w[2]) * aspect).takeIf { it != 0.0 } ?: 1.0
            val p = 2 * n + 2 + slot * 3
            out[p] = (w[0] - 0.5) * aspect
            out[p + 1] = w[1] - 0.5
            out[p + 2] = ln(size / 0.1)
        }
    }
    return out
}

private operator fun DoubleArray.component1() = this[0]
private operator fun DoubleArray.component2() = this[1]
private operator fun DoubleArray.component3() = this[2]

/**
 * Keeps the last [size] frames, or with [durationMs] set the last durationMs of frames (plus the one frame
 * just before), whatever the camera's frame rate.
 */
class FrameWindow(var size: Int = 30, var durationMs: Double = 0.0) {
    val frames = ArrayDeque<HandFrame>()

    fun push(frame: HandFrame) {
        frames.addLast(frame)
        if (durationMs > 0) {
            while (frames.size > 1 && frames[1].t <= frame.t - durationMs) frames.removeFirst()
        } else if (frames.size > size) frames.removeFirst()
    }

    val full: Boolean
        get() = if (durationMs > 0) frames.size > 1 && frames.last().t - frames.first().t >= durationMs
        else frames.size == size

    fun clear() = frames.clear()
}

/**
 * When the hands touch or cross, the tracker often labels both 'Left' or both 'Right', and one hand would be
 * lost. push() gives each hand its own label: hands keep the slot whose last position is nearest, or with no
 * history the more confident hand keeps its label. Labels that differ are never changed.
 */
class HandLabeler {
    // Wrist [x, y, z] where each slot (Left, Right) was last seen.
    private val last = arrayOfNulls<DoubleArray>(2)

    fun reset() {
        last[0] = null
        last[1] = null
    }

    fun push(frame: HandFrame): HandFrame {
        val aspect = aspectOf(frame)
        val hands = frame.hands.filter { it.landmarks.size == NUM_LANDMARKS }.take(2)
        val slots = assign(hands, aspect)
        hands.forEachIndexed { i, h -> last[slots[i]] = h.landmarks[WRIST] }
        if (hands.indices.all { hands[it].handedness == SLOT_NAMES[slots[it]] }) return frame
        return HandFrame(frame.t, frame.aspect, hands.mapIndexed { i, h -> h.withHandedness(SLOT_NAMES[slots[i]]) })
    }

    private fun dist(h: Hand, s: Int, aspect: Double): Double {
        val p = last[s] ?: return Double.POSITIVE_INFINITY
        val w = h.landmarks[WRIST]
        return hypot((w[0] - p[0]) * aspect, w[1] - p[1])
    }

    /** Slot (0 Left, 1 Right) for each hand. */
    private fun assign(hands: List<Hand>, aspect: Double): IntArray {
        val labels = hands.map { SLOT_NAMES.indexOf(it.handedness) }
        if (hands.size == 1) {
            val slot = if (labels[0] >= 0) labels[0] else if (dist(hands[0], 0, aspect) < dist(hands[0], 1, aspect)) 0 else 1
            return intArrayOf(slot)
        }
        if (hands.size < 2) return IntArray(0)
        val (a, b) = labels
        if (a >= 0 && b >= 0 && a != b) return intArrayOf(a, b)
        if (last[0] != null && last[1] != null) {
            val same = dist(hands[0], 0, aspect) + dist(hands[1], 1, aspect)
            val swapped = dist(hands[0], 1, aspect) + dist(hands[1], 0, aspect)
            return if (same <= swapped) intArrayOf(0, 1) else intArrayOf(1, 0)
        }
        val keep = if (hands[0].score >= hands[1].score) 0 else 1
        val s = if (labels[keep] >= 0) labels[keep] else 1
        return if (keep == 0) intArrayOf(s, 1 - s) else intArrayOf(1 - s, s)
    }
}
