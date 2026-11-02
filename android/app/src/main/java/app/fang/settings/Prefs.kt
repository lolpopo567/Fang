package app.fang.settings

import android.content.Context
import android.content.SharedPreferences
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonPrimitive
import kotlin.properties.ReadWriteProperty
import kotlin.reflect.KProperty

/**
 * App settings, saved in SharedPreferences under the same names as the web app's localStorage keys.
 * Each one is Compose state, so screens update as soon as a setting changes.
 */
class Prefs private constructor(private val sp: SharedPreferences) {
    var uiLang by string("fang.uiLang", "th")
    var theme by string("fang.theme", "auto")
    var captionLang by string("fang.captionLang", "th-TH")
    var captionSize by float("fang.captionSize", 34f)
    var captionOffline by bool("fang.captionOffline", false)
    var speechRate by float("fang.speechRate", 0.95f)
    var signLang by string("fang.signLang", "tsl")
    var signMode by string("fang.signMode", "offline")
    var signEndpoint by string("fang.signEndpoint", "")
    var signer by string("fang.signer", "")
    var phrases by stringList("fang.phrases")

    private fun string(key: String, default: String) = pref(sp.getString(key, default) ?: default) { putString(key, it) }
    private fun float(key: String, default: Float) = pref(sp.getFloat(key, default)) { putFloat(key, it) }
    private fun bool(key: String, default: Boolean) = pref(sp.getBoolean(key, default)) { putBoolean(key, it) }
    private fun stringList(key: String) = pref(
        runCatching { Json.parseToJsonElement(sp.getString(key, "[]")!!).jsonArray.map { it.jsonPrimitive.content } }
            .getOrDefault(emptyList()),
    ) { putString(key, JsonArray(it.map(::JsonPrimitive)).toString()) }

    private fun <T> pref(initial: T, write: SharedPreferences.Editor.(T) -> Unit) = object : ReadWriteProperty<Any?, T> {
        var state by mutableStateOf(initial)
        override fun getValue(thisRef: Any?, property: KProperty<*>) = state
        override fun setValue(thisRef: Any?, property: KProperty<*>, value: T) {
            state = value
            sp.edit().apply { write(value) }.apply()
        }
    }

    companion object {
        @Volatile private var instance: Prefs? = null

        fun get(context: Context): Prefs = instance ?: synchronized(this) {
            instance ?: Prefs(context.applicationContext.getSharedPreferences("fang", Context.MODE_PRIVATE)).also { instance = it }
        }
    }
}
