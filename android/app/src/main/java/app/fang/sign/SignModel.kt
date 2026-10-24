// Kotlin port of the web app's js/sign/model.js: the on-device sign classifier, a small MLP (or an average of
// several) whose weights come from training/train.py as JSON. The same model files serve both apps.
package app.fang.sign

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.boolean
import kotlinx.serialization.json.double
import kotlinx.serialization.json.doubleOrNull
import kotlinx.serialization.json.int
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import java.util.Base64
import kotlin.math.exp
import kotlin.math.floor
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.sqrt

const val MODEL_FORMAT = "fang-mlp-v1"
const val NO_HANDS = "_none"
private const val MIN_HAND_FRAMES = 0.3
private const val SEGMENTS = 4

/**
 * Picks [n] frames evenly spread over the last n / fps seconds, so a model sees the same time span whether the
 * phone tracks hands at 8 or 30 frames a second. For target time t it takes the first frame at or after t.
 */
fun resampleByTime(frames: List<HandFrame>, n: Int, fps: Double): List<HandFrame> {
    if (frames.isEmpty()) return frames
    val span = 1000.0 * n / fps
    val end = frames.last().t
    val out = ArrayList<HandFrame>(n)
    var i = 0
    for (k in 0 until n) {
        val target = end - span + (k + 0.5) * span / n
        while (i < frames.size - 1 && frames[i].t < target) i++
        out.add(frames[i])
    }
    return out
}

/** The same frame signed with the other hand leading: swap the two hand slots and flip x. */
fun mirrorVector(v: DoubleArray): DoubleArray {
    val n = NUM_LANDMARKS * 3
    val out = DoubleArray(v.size)
    System.arraycopy(v, n, out, 0, n)
    System.arraycopy(v, 0, out, n, n)
    out[2 * n] = v[2 * n + 1]
    out[2 * n + 1] = v[2 * n]
    for (i in 0 until 2 * n step 3) out[i] = -out[i]
    if (v.size > 2 * n + 2) {
        val p = 2 * n + 2
        out[p] = -v[p + 3]; out[p + 1] = v[p + 4]; out[p + 2] = v[p + 5]
        out[p + 3] = -v[p]; out[p + 4] = v[p + 1]; out[p + 5] = v[p + 2]
    }
    return out
}

/** True when the 'Left' slot hand leads the window: it is the only hand seen, or its wrist moves more. */
fun mainHandIsLeft(frames: List<HandFrame>): Boolean {
    val present = IntArray(2)
    val moved = DoubleArray(2)
    var prev = arrayOfNulls<DoubleArray>(2)
    for (f in frames) {
        val aspect = if (f.aspect == 0.0 || f.aspect.isNaN()) 1.0 else f.aspect
        val wrist = arrayOfNulls<DoubleArray>(2)
        for (hand in f.hands) {
            val slot = when (hand.handedness) {
                "Left" -> 0
                "Right" -> 1
                else -> continue
            }
            if (hand.landmarks.size != NUM_LANDMARKS) continue
            wrist[slot] = doubleArrayOf(hand.landmarks[0][0] * aspect, hand.landmarks[0][1])
        }
        for (s in 0..1) {
            val w = wrist[s] ?: continue
            present[s]++
            prev[s]?.let { moved[s] += hypot(w[0] - it[0], w[1] - it[1]) }
        }
        prev = wrist
    }
    if (present[0] != present[1] && minOf(present[0], present[1]) == 0) return present[0] > present[1]
    return moved[0] > moved[1]
}

/**
 * Summarizes a window of frames into one vector: mean, std, and mean(second half) - mean(first half) of the
 * frame vectors; version 2 adds the mean of each quarter of the window.
 */
fun sequenceFeatures(
    frames: List<HandFrame>,
    version: Int = 1,
    canonicalHand: Boolean = false,
    handPosition: Boolean = false,
    mirrored: Boolean = false,
): DoubleArray {
    var vecs = frames.map { frameVector(it, handPosition) }
    if (canonicalHand && mainHandIsLeft(frames)) vecs = vecs.map(::mirrorVector)
    if (mirrored) vecs = vecs.map(::mirrorVector)
    val n = vecs.size
    val d = vecs.firstOrNull()?.size ?: if (handPosition) 134 else 128
    val parts = 3 + if (version >= 2) SEGMENTS else 0
    val out = DoubleArray(parts * d)
    if (n == 0) return out
    val half = (n / 2).takeIf { it != 0 } ?: 1
    for (j in 0 until d) {
        var s = 0.0
        for (i in 0 until n) s += vecs[i][j]
        val m = s / n
        var v = 0.0
        for (i in 0 until n) { val e = vecs[i][j] - m; v += e * e }
        var a = 0.0
        for (i in 0 until half) a += vecs[i][j]
        var b = 0.0
        for (i in half until n) b += vecs[i][j]
        out[j] = m
        out[d + j] = sqrt(v / n)
        out[2 * d + j] = if (n - half > 0) b / (n - half) - a / half else 0.0
        if (version >= 2) {
            for (k in 0 until SEGMENTS) {
                val lo = floor(k.toDouble() * n / SEGMENTS).toInt()
                val hi = max(floor((k + 1).toDouble() * n / SEGMENTS).toInt(), lo + 1)
                var c = 0.0
                for (i in lo until hi) c += vecs[i][j]
                out[(3 + k) * d + j] = c / (hi - lo)
            }
        }
    }
    return out
}

data class Ranked(val label: String, val text: String, val confidence: Double)

data class ModelOutput(val label: String, val text: String, val confidence: Double, val alternatives: List<Ranked>)

private class Layer(val w: Array<FloatArray>, val b: DoubleArray)

class SignModel private constructor(json: JsonObject) {
    val labels: List<String>
    val display: Map<String, String>
    val window: Int
    val fps: Double?
    private val features: Int
    private val canonicalHand: Boolean
    private val handPosition: Boolean
    private val skipEmptyFrames: Boolean
    private val mirrorAverage: Boolean
    private val mean: DoubleArray
    private val std: DoubleArray
    private val nets: List<List<Layer>>

    init {
        val format = json["format"]?.jsonPrimitive?.content
        require(format == MODEL_FORMAT) { "unsupported model format $format" }
        fun flag(key: String) = json[key]?.jsonPrimitive?.boolean ?: false
        labels = json["labels"]!!.jsonArray.map { it.jsonPrimitive.content }
        display = json["display"]?.jsonObject?.mapValues { it.value.jsonPrimitive.content } ?: emptyMap()
        window = json["window"]!!.jsonPrimitive.int
        fps = json["fps"]?.jsonPrimitive?.doubleOrNull?.takeIf { it > 0 }
        features = json["features"]?.jsonPrimitive?.int ?: 1
        canonicalHand = flag("canonicalHand")
        handPosition = flag("handPosition")
        skipEmptyFrames = flag("skipEmptyFrames")
        mirrorAverage = flag("mirrorAverage")
        mean = json["mean"]!!.jsonArray.toDoubles()
        std = json["std"]!!.jsonArray.toDoubles()
        // `ensemble` [{layers}, ...] averages several nets; `layers` [{W, b}, ...] is one net with ReLU between
        // layers; older models have one hidden layer as W1/b1/W2/b2.
        val netJson: List<JsonArray> = json["ensemble"]?.jsonArray?.map { it.jsonObject["layers"]!!.jsonArray }
            ?: listOf(
                json["layers"]?.jsonArray ?: JsonArray(
                    listOf(
                        JsonObject(mapOf("W" to json["W1"]!!, "b" to json["b1"]!!)),
                        JsonObject(mapOf("W" to json["W2"]!!, "b" to json["b2"]!!)),
                    ),
                ),
            )
        nets = netJson.map { layers ->
            layers.map { l -> Layer(decodeMatrix(l.jsonObject["W"]!!), l.jsonObject["b"]!!.jsonArray.toDoubles()) }
        }
    }

    /** Ranked guesses with softmax probabilities; text is the word to show. */
    fun predict(input: List<HandFrame>): ModelOutput {
        var frames = fps?.let { resampleByTime(input, window, it) } ?: input
        // The model never saw empty windows, so without hands in view it would still name a sign.
        val withHands = frames.filter { it.hands.isNotEmpty() }
        if (withHands.size < MIN_HAND_FRAMES * frames.size) return ModelOutput(NO_HANDS, NO_HANDS, 1.0, emptyList())
        // Models trained on frames where the tracker often missed the hands (PopSign) skip hand-less frames.
        if (skipEmptyFrames) frames = withHands
        // mirrorAverage: also score the window as a left-handed signer would make it, and average.
        val views = (if (mirrorAverage) listOf(false, true) else listOf(false)).map { mirrored ->
            val f = sequenceFeatures(frames, features, canonicalHand, handPosition, mirrored)
            DoubleArray(f.size) { (f[it] - mean[it]) / std[it] }
        }
        val probs = DoubleArray(labels.size)
        val parts = nets.size * views.size
        for (layers in nets) {
            for (x in views) {
                val p = softmax(forward(layers, x))
                for (c in p.indices) probs[c] += p[c] / parts
            }
        }
        val ranked = probs.indices
            .map { Ranked(labels[it], display[labels[it]] ?: labels[it], probs[it]) }
            .sortedByDescending { it.confidence }
        val top = ranked[0]
        return ModelOutput(top.label, top.text, top.confidence, ranked.subList(1, minOf(4, ranked.size)))
    }

    companion object {
        fun parse(text: String) = SignModel(Json.parseToJsonElement(text).jsonObject)
    }
}

private fun JsonArray.toDoubles() = DoubleArray(size) { this[it].jsonPrimitive.double }

/**
 * A weight matrix from the model JSON: plain rows, or the compact int8 form from training/train.py
 * encode_matrix, { q8: base64 int8 row-major, rows, cols, scale: per-row }, where W[r][c] = q8 * scale[r].
 */
private fun decodeMatrix(w: JsonElement): Array<FloatArray> {
    if (w is JsonArray) return Array(w.size) { r -> w[r].jsonArray.let { row -> FloatArray(row.size) { row[it].jsonPrimitive.double.toFloat() } } }
    val o = w.jsonObject
    val q = Base64.getDecoder().decode(o["q8"]!!.jsonPrimitive.content)
    val rows = o["rows"]!!.jsonPrimitive.int
    val cols = o["cols"]!!.jsonPrimitive.int
    val scale = o["scale"]!!.jsonArray.toDoubles()
    return Array(rows) { r -> FloatArray(cols) { c -> (q[r * cols + c] * scale[r]).toFloat() } }
}

private fun forward(layers: List<Layer>, input: DoubleArray): DoubleArray {
    var h = input
    layers.forEachIndexed { l, layer ->
        val next = DoubleArray(layer.b.size)
        for (k in next.indices) {
            val row = layer.w[k]
            var s = 0.0
            for (i in h.indices) s += row[i] * h[i]
            next[k] = layer.b[k] + s
            if (l < layers.size - 1 && next[k] < 0) next[k] = 0.0
        }
        h = next
    }
    return h
}

private fun softmax(logits: DoubleArray): DoubleArray {
    val max = logits.max()
    val exps = DoubleArray(logits.size) { exp(logits[it] - max) }
    val sum = exps.sum()
    return DoubleArray(exps.size) { exps[it] / sum }
}
