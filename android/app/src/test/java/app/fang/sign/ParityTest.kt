package app.fang.sign

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.boolean
import kotlinx.serialization.json.double
import kotlinx.serialization.json.int
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import org.junit.Assert.assertEquals
import org.junit.Test
import java.io.File
import kotlin.math.abs

/**
 * The Kotlin sign pipeline must give the same numbers as the web app's, because both run the same trained
 * models. parity.json holds clips and the JS outputs for them (tools/make-parity-fixture.mjs).
 */
class ParityTest {
    private val cases = Json.parseToJsonElement(javaClass.getResource("/parity.json")!!.readText())
        .jsonObject["cases"]!!.jsonArray.map { it.jsonObject }

    private fun frames(c: JsonObject) = c["raw"]!!.jsonArray.map { f ->
        val o = f.jsonObject
        HandFrame(
            o["t"]!!.jsonPrimitive.double,
            o["aspect"]!!.jsonPrimitive.double,
            o["hands"]!!.jsonArray.map { h ->
                val ho = h.jsonObject
                Hand(
                    ho["handedness"]!!.jsonPrimitive.content,
                    ho["score"]!!.jsonPrimitive.double,
                    ho["landmarks"]!!.jsonArray.map { p -> p.jsonArray.map { it.jsonPrimitive.double }.toDoubleArray() },
                )
            },
        )
    }

    private fun labeled(c: JsonObject): List<HandFrame> {
        val labeler = HandLabeler()
        return frames(c).map(labeler::push)
    }

    private fun assertClose(expected: List<Double>, actual: DoubleArray, tol: Double, what: String) {
        assertEquals("$what length", expected.size, actual.size)
        expected.forEachIndexed { i, e -> assert(abs(e - actual[i]) <= tol) { "$what[$i]: expected $e, got ${actual[i]}" } }
    }

    private fun doubles(e: kotlinx.serialization.json.JsonElement) = e.jsonArray.map { it.jsonPrimitive.double }

    @Test
    fun handLabelerMatches() {
        for (c in cases) {
            val expected = c["labeled"]!!.jsonArray.map { f -> f.jsonArray.map { it.jsonPrimitive.content } }
            assertEquals(c["kind"].toString(), expected, labeled(c).map { f -> f.hands.map { it.handedness } })
        }
    }

    @Test
    fun frameVectorsMatch() {
        for (c in cases) {
            val frames = labeled(c)
            listOf(0, 7, 20).forEachIndexed { k, i ->
                assertClose(doubles(c["frameVectors"]!!.jsonArray[k]), frameVector(frames[i], true), 1e-12, "${c["kind"]} frame $i")
            }
        }
    }

    @Test
    fun resampleMatches() {
        for (c in cases) {
            assertClose(doubles(c["resampledT"]!!), resampleByTime(labeled(c), 24, 15.0).map { it.t }.toDoubleArray(), 0.0, "resample")
        }
    }

    @Test
    fun sequenceFeaturesMatch() {
        for (c in cases) {
            val frames = labeled(c)
            for (s in c["sequence"]!!.jsonArray.map { it.jsonObject }) {
                val out = sequenceFeatures(
                    frames,
                    version = s["version"]!!.jsonPrimitive.int,
                    canonicalHand = s["canonicalHand"]!!.jsonPrimitive.boolean,
                    handPosition = s["handPosition"]!!.jsonPrimitive.boolean,
                    mirrored = s["mirrored"]!!.jsonPrimitive.boolean,
                )
                assertClose(doubles(s["out"]!!), out, 1e-9, "${c["kind"]} $s".take(80))
            }
        }
    }

    @Test
    fun modelPredictionsMatch() {
        for (name in listOf("tsl", "asl")) {
            val model = SignModel.parse(File("../../models/$name-model.json").readText())
            for (c in cases) {
                val expected = c["predictions"]!!.jsonObject[name]!!.jsonObject
                val p = model.predict(labeled(c))
                val what = "$name ${c["kind"]}"
                assertEquals(what, expected["label"]!!.jsonPrimitive.content, p.label)
                // Weights are stored as float here and double in the browser: agree to well under 0.01%.
                assertEquals(what, expected["confidence"]!!.jsonPrimitive.double, p.confidence, 1e-6)
                val alts = expected["alternatives"]!!.jsonArray.map { it.jsonObject }
                assertEquals(what, alts.map { it["label"]!!.jsonPrimitive.content }, p.alternatives.map { it.label })
                alts.forEachIndexed { i, a -> assertEquals(what, a["confidence"]!!.jsonPrimitive.double, p.alternatives[i].confidence, 1e-6) }
            }
        }
    }

    @Test
    fun noHandsGivesNoSign() {
        val model = SignModel.parse(File("../../models/tsl-model.json").readText())
        val empty = (0 until 40).map { HandFrame(1000.0 + it * 66, 0.75, emptyList()) }
        assertEquals(NO_HANDS, model.predict(empty).label)
    }
}
