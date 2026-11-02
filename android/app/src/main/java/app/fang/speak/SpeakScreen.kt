package app.fang.speak

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.OutlinedTextFieldDefaults
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.window.Dialog
import androidx.compose.ui.window.DialogProperties
import androidx.lifecycle.viewmodel.compose.viewModel
import app.fang.settings.Prefs
import app.fang.ui.FangCard
import app.fang.ui.LocalLang
import app.fang.ui.OutlineButton
import app.fang.ui.PrimaryButton
import app.fang.ui.ScreenHead
import app.fang.ui.SectionLabel
import app.fang.ui.StatusText
import app.fang.ui.Tone
import app.fang.ui.colors
import app.fang.ui.t
import androidx.compose.ui.platform.LocalContext

@Composable
fun SpeakScreen(vm: SpeakViewModel = viewModel()) {
    val prefs = Prefs.get(LocalContext.current)
    val lang = LocalLang.current
    Column {
        ScreenHead(Tone.Orange, "keyboard", t("nav.speak"), t("home.speakDesc"))
        FangCard(Modifier.fillMaxWidth().padding(vertical = 8.dp)) {
            Column(Modifier.padding(16.dp)) {
                OutlinedTextField(
                    value = vm.text,
                    onValueChange = { vm.text = it },
                    label = { Text(t("speak.label")) },
                    minLines = 3,
                    textStyle = MaterialTheme.typography.bodyLarge.copy(fontSize = 22.sp, lineHeight = 32.sp),
                    modifier = Modifier.fillMaxWidth(),
                    shape = RoundedCornerShape(14.dp),
                    colors = OutlinedTextFieldDefaults.colors(
                        focusedBorderColor = colors.orange, focusedLabelColor = colors.orange, cursorColor = colors.orange,
                        unfocusedContainerColor = colors.surface2, focusedContainerColor = colors.surface2,
                    ),
                )
                Spacer(Modifier.height(12.dp))
                Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                    PrimaryButton(t("speak.say"), "speaker", Tone.Orange, { vm.say(vm.text) }, Modifier.weight(1f))
                    OutlineButton(t("speak.show"), "expand", { vm.showBig(vm.text) }, Modifier.weight(1f))
                }
                OutlineButton(t("speak.savePhrase"), "bookmark", vm::savePhrase, Modifier.padding(top = 6.dp), tint = colors.orange, ghost = true)
            }
        }
        StatusText(if (vm.status.isEmpty()) "" else t(vm.status))
        SectionLabel(t("speak.phrases"))
        val all = prefs.phrases.map { it to true } + DEFAULT_PHRASES.getValue(lang).map { it to false }
        for ((p, custom) in all) {
            PhraseRow(p, custom, onSay = { vm.text = p; vm.say(p) }, onShow = { vm.showBig(p) }, onDelete = { vm.deletePhrase(p) })
        }
    }
    vm.big?.let { BigText(it) { vm.big = null } }
}

@Composable
private fun PhraseRow(phrase: String, custom: Boolean, onSay: () -> Unit, onShow: () -> Unit, onDelete: () -> Unit) {
    FangCard(Modifier.fillMaxWidth().padding(vertical = 5.dp)) {
        Column {
            Text(
                phrase,
                style = MaterialTheme.typography.bodyLarge.copy(fontWeight = FontWeight.SemiBold),
                modifier = Modifier
                    .fillMaxWidth()
                    .clickable(role = Role.Button, onClick = onSay)
                    .padding(start = 16.dp, end = 16.dp, top = 14.dp, bottom = 6.dp),
            )
            Row(Modifier.padding(start = 8.dp, end = 8.dp, bottom = 8.dp), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                OutlineButton(t("speak.say"), "speaker", onSay, small = true, ghost = true, tint = colors.orange)
                OutlineButton(t("speak.show"), "expand", onShow, small = true, ghost = true, tint = colors.muted)
                if (custom) OutlineButton(t("speak.delete"), "trash", onDelete, small = true, ghost = true, tint = colors.muted)
            }
        }
    }
}

/** Full-screen large text to show someone, like the web app's "show large" card. */
@Composable
fun BigText(text: String, onClose: () -> Unit) {
    Dialog(onDismissRequest = onClose, properties = DialogProperties(usePlatformDefaultWidth = false, decorFitsSystemWindows = false)) {
        Box(Modifier.fillMaxSize().background(colors.surface).safeDrawingPadding()) {
            Column(Modifier.fillMaxSize().padding(24.dp)) {
                Box(Modifier.weight(1f).fillMaxWidth().verticalScroll(rememberScrollState()), contentAlignment = Alignment.Center) {
                    Text(
                        text,
                        style = MaterialTheme.typography.headlineMedium.copy(fontSize = 52.sp, lineHeight = 68.sp),
                        textAlign = TextAlign.Center,
                        modifier = Modifier.fillMaxWidth().padding(vertical = 24.dp),
                    )
                }
                OutlineButton(t("speak.close"), "close", onClose, Modifier.fillMaxWidth().clip(RoundedCornerShape(18.dp)))
            }
        }
    }
}
