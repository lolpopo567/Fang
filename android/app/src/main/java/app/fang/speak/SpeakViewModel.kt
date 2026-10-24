package app.fang.speak

import android.app.Application
import android.speech.tts.TextToSpeech
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.AndroidViewModel
import app.fang.settings.Prefs
import java.util.Locale

// Same quick phrases as the web app's js/speak.js.
val DEFAULT_PHRASES = mapOf(
    "th" to listOf(
        "ฉันหูหนวก/หูตึง กรุณาพิมพ์หรือเขียนให้อ่าน",
        "กรุณาพูดช้าๆ และหันหน้ามาทางฉัน",
        "ช่วยเขียนให้หน่อยได้ไหม",
        "ขอบคุณครับ/ค่ะ",
        "ห้องน้ำอยู่ที่ไหน",
        "ฉันต้องการความช่วยเหลือ",
        "ช่วยเรียกรถพยาบาล (1669)",
    ),
    "en" to listOf(
        "I am deaf / hard of hearing. Please type or write it down.",
        "Please speak slowly and face me.",
        "Could you write that down?",
        "Thank you.",
        "Where is the restroom?",
        "I need help.",
        "Please call an ambulance.",
    ),
)

private val THAI = Regex("[฀-๿]")

/** Type to speak with Android's TextToSpeech. Without a voice for the text's language it shows the text large. */
class SpeakViewModel(app: Application) : AndroidViewModel(app) {
    private val prefs = Prefs.get(app)
    private var ready = false
    private val tts = TextToSpeech(app) { ready = it == TextToSpeech.SUCCESS }

    var text by mutableStateOf("")
    /** i18n key or "". */
    var status by mutableStateOf("")
    /** Text shown full screen, or null. */
    var big by mutableStateOf<String?>(null)

    fun say(phrase: String) {
        if (phrase.isBlank()) return run { status = "speak.empty" }
        // Speak in the script of the text, so Thai text gets a Thai voice whatever the UI language.
        val locale = if (THAI.containsMatchIn(phrase)) Locale.forLanguageTag("th-TH") else Locale.US
        if (!ready || tts.isLanguageAvailable(locale) < TextToSpeech.LANG_AVAILABLE) {
            status = "speak.noVoice"
            big = phrase
            return
        }
        tts.language = locale
        tts.setSpeechRate(prefs.speechRate)
        tts.speak(phrase, TextToSpeech.QUEUE_FLUSH, null, "fang")
        status = ""
    }

    fun showBig(phrase: String) {
        if (phrase.isBlank()) status = "speak.empty" else big = phrase
    }

    fun savePhrase() {
        val p = text.trim()
        if (p.isEmpty()) return run { status = "speak.empty" }
        if (p !in prefs.phrases) prefs.phrases = listOf(p) + prefs.phrases
        status = "speak.saved"
    }

    fun deletePhrase(p: String) {
        prefs.phrases = prefs.phrases - p
    }

    fun testVoice(lang: String) = say(DEFAULT_PHRASES.getValue(lang).first())

    override fun onCleared() {
        tts.shutdown()
    }
}
