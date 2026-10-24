package app.fang.settings

import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Slider
import androidx.compose.material3.SliderDefaults
import androidx.compose.material3.Switch
import androidx.compose.material3.SwitchDefaults
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import app.fang.R
import app.fang.ui.FangCard
import app.fang.ui.FinePrint
import app.fang.ui.GradientTile
import app.fang.ui.OutlineButton
import app.fang.ui.PlexThai
import app.fang.ui.ScreenHead
import app.fang.ui.SelectField
import app.fang.ui.Tone
import app.fang.sign.signLanguageOptions
import app.fang.ui.colors
import app.fang.ui.t

/**
 * Settings, grouped like the web app's. The callbacks let running features pick up a change straight away.
 */
@Composable
fun SettingsScreen(onSpeechChanged: () -> Unit, onSignLangChanged: (String) -> Unit, onSignChanged: () -> Unit, onTestVoice: () -> Unit) {
    val prefs = Prefs.get(LocalContext.current)
    Column {
        ScreenHead(Tone.Slate, "settings", t("nav.settings"), t("home.settingsDesc"))

        Group(Tone.Slate, "palette", t("settings.display")) {
            SelectField(t("ui.language"), listOf("th" to "ไทย", "en" to "English"), prefs.uiLang, { prefs.uiLang = it })
            SelectField(
                t("settings.theme"),
                listOf("auto" to t("settings.themeAuto"), "light" to t("settings.themeLight"), "dark" to t("settings.themeDark")),
                prefs.theme, { prefs.theme = it },
            )
        }

        Group(Tone.Blue, "captions", t("nav.captions")) {
            SelectField(t("captions.speechLang"), listOf("th-TH" to "ไทย (th-TH)", "en-US" to "English (en-US)"), prefs.captionLang, {
                prefs.captionLang = it
                onSpeechChanged()
            })
            Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.padding(top = 8.dp)) {
                Text(t("captions.offline"), style = MaterialTheme.typography.bodyLarge, modifier = Modifier.weight(1f))
                Switch(
                    checked = prefs.captionOffline,
                    onCheckedChange = { prefs.captionOffline = it; onSpeechChanged() },
                    colors = SwitchDefaults.colors(checkedTrackColor = colors.blue),
                )
            }
            FinePrint(t("captions.offlineHelp"))
            Text(t("captions.textSize"), style = MaterialTheme.typography.labelMedium, color = colors.muted)
            Slider(
                prefs.captionSize, { prefs.captionSize = it }, valueRange = 20f..72f,
                colors = SliderDefaults.colors(thumbColor = colors.blue, activeTrackColor = colors.blue),
            )
            Text(
                t("settings.preview"),
                fontFamily = PlexThai,
                fontSize = prefs.captionSize.sp,
                lineHeight = (prefs.captionSize * 1.3f).sp,
                maxLines = 2,
                overflow = TextOverflow.Ellipsis,
                color = Color.White,
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(top = 4.dp)
                    .clip(RoundedCornerShape(14.dp))
                    .background(Color.Black)
                    .padding(horizontal = 16.dp, vertical = 12.dp),
            )
        }

        Group(Tone.Orange, "keyboard", t("nav.speak")) {
            Text("${t("settings.speechRate")} ${"%.2f".format(prefs.speechRate)}×", style = MaterialTheme.typography.labelMedium, color = colors.muted)
            Slider(
                prefs.speechRate, { prefs.speechRate = (it * 20).let(Math::round) / 20f }, valueRange = 0.5f..1.5f,
                colors = SliderDefaults.colors(thumbColor = colors.orange, activeTrackColor = colors.orange),
            )
            OutlineButton(t("settings.testVoice"), "speaker", onTestVoice)
        }

        Group(Tone.Violet, "hand", t("nav.sign")) {
            SelectField(t("sign.language"), signLanguageOptions(), prefs.signLang, {
                prefs.signLang = it
                onSignLangChanged(it)
            })
            SelectField(
                t("sign.mode"),
                listOf("offline" to t("sign.modeOffline"), "online" to t("sign.modeOnline"), "hybrid" to t("sign.modeHybrid")),
                prefs.signMode, { prefs.signMode = it; onSignChanged() },
            )
            if (prefs.signMode != "offline") {
                OutlinedTextField(
                    prefs.signEndpoint, { prefs.signEndpoint = it.trim(); onSignChanged() },
                    label = { Text(t("sign.endpoint")) },
                    placeholder = { Text("https://example.org/api/recognize") },
                    singleLine = true,
                    modifier = Modifier.fillMaxWidth().padding(top = 6.dp),
                )
            }
        }

        FangCard(Modifier.fillMaxWidth().padding(vertical = 10.dp)) {
            Row(Modifier.padding(16.dp), verticalAlignment = Alignment.CenterVertically) {
                Image(painterResource(R.drawable.kitten), null, Modifier.size(84.dp))
                Spacer(Modifier.width(14.dp))
                Column {
                    Text("Fang ฟัง", style = MaterialTheme.typography.titleMedium)
                    Text(t("settings.about"), style = MaterialTheme.typography.bodyMedium, color = colors.muted)
                }
            }
        }
    }
}

@Composable
private fun Group(tone: Tone, iconName: String, title: String, content: @Composable () -> Unit) {
    FangCard(Modifier.fillMaxWidth().padding(vertical = 8.dp)) {
        Column(Modifier.padding(16.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.padding(bottom = 6.dp)) {
                GradientTile(tone, iconName, size = 40.dp)
                Spacer(Modifier.width(12.dp))
                Text(title, style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.Bold))
            }
            content()
        }
    }
}
