package app.fang.sign

import android.content.Context
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.add
import kotlinx.serialization.json.addJsonObject
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.put
import kotlinx.serialization.json.putJsonArray
import java.io.File
import java.io.OutputStream
import java.time.Instant
import kotlin.math.roundToLong

private fun round4(n: Double) = (n * 1e4).roundToLong() / 1e4

/**
 * Labelled 2-second landmark samples for training, kept in app storage and exported as the web app's
 * `fang-handframes-v1` JSON, so samples from both apps go into the same training scripts. No video is stored.
 */
class SampleRecorder(context: Context) {
    private val file = File(context.filesDir, "sign-samples.json")
    private val samples: MutableList<JsonElement> = runCatching {
        Json.parseToJsonElement(file.readText()).jsonObject["samples"]!!.jsonArray.toMutableList()
    }.getOrDefault(mutableListOf())

    val count get() = samples.size

    /** Adds a sample and returns the new count. */
    fun add(label: String, signer: String, frames: List<HandFrame>): Int {
        val t0 = frames.first().t
        samples.add(
            buildJsonObject {
                put("label", label)
                if (signer.isNotEmpty()) put("signer", signer)
                put("recordedAt", Instant.now().toString())
                put("source", "android")
                putJsonArray("frames") {
                    for (f in frames) addJsonObject {
                        put("t", (f.t - t0).roundToLong())
                        put("aspect", round4(f.aspect))
                        // 4 decimals keeps sub-pixel precision and makes samples ~3x smaller.
                        putJsonArray("hands") {
                            for (h in f.hands) addJsonObject {
                                put("handedness", h.handedness)
                                put("score", round4(h.score))
                                putJsonArray("landmarks") {
                                    for (p in h.landmarks) add(JsonArray(p.map { JsonPrimitive(round4(it)) }))
                                }
                            }
                        }
                    }
                }
            },
        )
        save()
        return samples.size
    }

    fun clear() {
        samples.clear()
        save()
    }

    fun export(out: OutputStream) = out.write(json().toByteArray())

    private fun json() = buildJsonObject {
        put("format", "fang-handframes-v1")
        put("samples", JsonArray(samples))
    }.toString()

    private fun save() = file.writeText(json())
}
