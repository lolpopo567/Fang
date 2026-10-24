// Recognizer interface, as in the web app's js/sign/recognizer.js. Every recognizer takes a window of
// HandFrames and returns a Prediction. Confidence is the model's own score, NOT a guarantee of correctness.
package app.fang.sign

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.add
import kotlinx.serialization.json.addJsonObject
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.doubleOrNull
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.put
import kotlinx.serialization.json.putJsonArray
import java.net.HttpURLConnection
import java.net.SocketTimeoutException
import java.net.URL

const val UNCERTAIN_BELOW = 0.6

enum class Status { OK, NO_MODEL, NOT_CONFIGURED, ERROR }

data class Prediction(
    val status: Status,
    val source: String,
    val label: String? = null,
    val text: String? = null,
    val confidence: Double = 0.0,
    val alternatives: List<Ranked> = emptyList(),
    val error: String? = null,
) {
    val uncertain get() = status == Status.OK && confidence < UNCERTAIN_BELOW
}

interface Recognizer {
    val available: Boolean
    suspend fun predict(frames: List<HandFrame>): Prediction
}

class OfflineRecognizer(@Volatile var model: SignModel? = null) : Recognizer {
    override val available get() = model != null

    override suspend fun predict(frames: List<HandFrame>): Prediction {
        val m = model ?: return Prediction(Status.NO_MODEL, "offline")
        val r = withContext(Dispatchers.Default) { m.predict(frames) }
        return Prediction(Status.OK, "offline", r.label, r.text, r.confidence, r.alternatives)
    }
}

/**
 * Sends only landmark coordinates (never video) to a server.
 * Request:  POST { version: 1, frames: HandFrame[] }
 * Response: { label, confidence, alternatives? }
 */
class OnlineRecognizer(@Volatile var endpoint: String = "", private val timeoutMs: Int = 3000) : Recognizer {
    override val available get() = endpoint.isNotBlank()

    override suspend fun predict(frames: List<HandFrame>): Prediction {
        val url = endpoint.trim()
        if (url.isEmpty()) return Prediction(Status.NOT_CONFIGURED, "online")
        return withContext(Dispatchers.IO) {
            try {
                val conn = URL(url).openConnection() as HttpURLConnection
                conn.connectTimeout = timeoutMs
                conn.readTimeout = timeoutMs
                conn.requestMethod = "POST"
                conn.doOutput = true
                conn.setRequestProperty("Content-Type", "application/json")
                conn.outputStream.use { it.write(requestBody(frames).toByteArray()) }
                if (conn.responseCode !in 200..299) {
                    return@withContext Prediction(Status.ERROR, "online", error = "HTTP ${conn.responseCode}")
                }
                val body = Json.parseToJsonElement(conn.inputStream.bufferedReader().readText()).jsonObject
                val label = (body["label"] as? JsonPrimitive)?.takeIf { it.isString }?.content
                    ?: return@withContext Prediction(Status.ERROR, "online", error = "bad response")
                val alternatives = (body["alternatives"] as? JsonArray)?.mapNotNull { alt ->
                    val o = alt as? JsonObject ?: return@mapNotNull null
                    val l = o["label"]?.jsonPrimitive?.content ?: return@mapNotNull null
                    Ranked(l, l, o["confidence"]?.jsonPrimitive?.doubleOrNull ?: 0.0)
                } ?: emptyList()
                Prediction(
                    Status.OK, "online", label, label,
                    body["confidence"]?.jsonPrimitive?.doubleOrNull ?: 0.0, alternatives,
                )
            } catch (e: SocketTimeoutException) {
                Prediction(Status.ERROR, "online", error = "timeout")
            } catch (e: Exception) {
                Prediction(Status.ERROR, "online", error = e.message ?: e.javaClass.simpleName)
            }
        }
    }
}

fun requestBody(frames: List<HandFrame>): String = buildJsonObject {
    put("version", 1)
    putJsonArray("frames") {
        for (f in frames) addJsonObject {
            put("t", f.t)
            put("aspect", f.aspect)
            putJsonArray("hands") {
                for (h in f.hands) addJsonObject {
                    put("handedness", h.handedness)
                    put("score", h.score)
                    putJsonArray("landmarks") {
                        for (p in h.landmarks) add(JsonArray(p.map { JsonPrimitive(it) }))
                    }
                }
            }
        }
    }
}.toString()

enum class RecognizerMode(val key: String) {
    OFFLINE("offline"), ONLINE("online"), HYBRID("hybrid");

    companion object {
        fun of(key: String) = entries.firstOrNull { it.key == key } ?: OFFLINE
    }
}

/** Picks a recognizer per mode. Hybrid prefers on-device, falls back to online, then reports why neither worked. */
class RecognizerHub(val offline: OfflineRecognizer, val online: OnlineRecognizer) {
    @Volatile var mode = RecognizerMode.OFFLINE

    suspend fun predict(frames: List<HandFrame>): Prediction = when (mode) {
        RecognizerMode.OFFLINE -> offline.predict(frames)
        RecognizerMode.ONLINE -> online.predict(frames)
        RecognizerMode.HYBRID -> when {
            offline.available -> offline.predict(frames)
            online.available -> online.predict(frames)
            else -> Prediction(Status.NO_MODEL, "offline")
        }
    }
}
