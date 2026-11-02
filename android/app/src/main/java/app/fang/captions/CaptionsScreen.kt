package app.fang.captions

import android.Manifest
import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.content.pm.PackageManager
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Slider
import androidx.compose.material3.SliderDefaults
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalView
import androidx.compose.ui.semantics.LiveRegionMode
import androidx.compose.ui.semantics.liveRegion
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat
import androidx.lifecycle.viewmodel.compose.viewModel
import app.fang.settings.Prefs
import app.fang.ui.FinePrint
import app.fang.ui.Live
import app.fang.ui.Notice
import app.fang.ui.OutlineButton
import app.fang.ui.PlexThai
import app.fang.ui.PrimaryButton
import app.fang.ui.ScreenHead
import app.fang.ui.StatusText
import app.fang.ui.Tone
import app.fang.ui.colors
import app.fang.ui.t
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

private val CaptionBg = Color.Black
private val Interim = Color(0xFFA9C2FF)

@Composable
fun CaptionsScreen(vm: CaptionsViewModel = viewModel()) {
    val context = LocalContext.current
    val prefs = Prefs.get(context)
    val askMic = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        if (granted) vm.start() else vm.status = "captions.err.not-allowed"
    }
    val saveFile = rememberLauncherForActivityResult(ActivityResultContracts.CreateDocument("text/plain")) { uri ->
        if (uri != null) {
            runCatching {
                context.contentResolver.openOutputStream(uri)?.use { it.write(vm.transcript().toByteArray()) }
            }.onSuccess { vm.status = "captions.saved" }
        }
    }

    // Keep the screen on while captions are running.
    val view = LocalView.current
    DisposableEffect(vm.listening) {
        view.keepScreenOn = vm.listening
        onDispose { view.keepScreenOn = false }
    }

    Column {
        ScreenHead(Tone.Blue, "captions", t("nav.captions"), t("home.captionsDesc"))
        if (!vm.supported) {
            Notice(t("captions.unsupported"), Modifier.padding(vertical = 8.dp))
        }
        CaptionBox(vm.lines, vm.interim, vm.listening, prefs.captionSize)
        StatusText(statusText(vm.status, vm.statusDetail))
        PrimaryButton(
            text = t(if (vm.listening) "captions.stop" else "captions.start"),
            iconName = if (vm.listening) "stop" else "mic",
            tone = Tone.Blue,
            live = vm.listening,
            enabled = vm.supported,
            modifier = Modifier.fillMaxWidth(),
            onClick = {
                val granted = ContextCompat.checkSelfPermission(context, Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED
                if (vm.listening || granted) vm.toggle() else askMic.launch(Manifest.permission.RECORD_AUDIO)
            },
        )
        Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.padding(vertical = 8.dp)) {
            Text("ก", fontSize = 16.sp, color = colors.muted)
            Slider(
                value = prefs.captionSize,
                onValueChange = { prefs.captionSize = it },
                valueRange = 20f..72f,
                modifier = Modifier.weight(1f).padding(horizontal = 12.dp),
                colors = SliderDefaults.colors(thumbColor = colors.blue, activeTrackColor = colors.blue),
            )
            Text("ก", fontSize = 30.sp, color = colors.muted)
        }
        Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
            OutlineButton(t("captions.copy"), "copy", modifier = Modifier.weight(1f), small = true, onClick = {
                val cm = context.getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
                cm.setPrimaryClip(ClipData.newPlainText("captions", vm.transcript()))
                vm.status = "captions.copied"
            })
            OutlineButton(t("captions.save"), "download", modifier = Modifier.weight(1f), small = true, onClick = {
                val stamp = SimpleDateFormat("yyyy-MM-dd-HH-mm", Locale.ROOT).format(Date())
                saveFile.launch("captions-$stamp.txt")
            })
            OutlineButton(t("captions.clear"), "trash", modifier = Modifier.weight(1f), small = true, onClick = vm::clear)
        }
        Spacer(Modifier.height(8.dp))
        FinePrint(t("captions.privacy"), "lock")
    }
}

@Composable
private fun statusText(key: String, detail: String): String = when {
    key.isEmpty() -> ""
    key == "captions.err.generic" -> t(key) + detail
    else -> t(key)
}

/** White text on black, whatever the theme, at the user's chosen size. Newest line at the bottom. */
@Composable
fun CaptionBox(lines: List<String>, interim: String, live: Boolean, size: Float, modifier: Modifier = Modifier) {
    val scroll = rememberScrollState()
    LaunchedEffect(lines.size, interim) { scroll.animateScrollTo(scroll.maxValue) }
    val shape = RoundedCornerShape(20.dp)
    val style = TextStyle(fontFamily = PlexThai, fontSize = size.sp, lineHeight = (size * 1.35f).sp)
    Column(
        modifier
            .fillMaxWidth()
            .heightIn(min = 280.dp, max = 460.dp)
            .clip(shape)
            .background(CaptionBg)
            .border(if (live) 3.dp else 0.dp, if (live) Live else Color.Transparent, shape)
            .verticalScroll(scroll)
            .padding(20.dp)
            .semantics { liveRegion = LiveRegionMode.Polite },
    ) {
        if (lines.isEmpty() && interim.isEmpty()) {
            Text(t("captions.placeholder"), style = style.copy(fontSize = 20.sp, lineHeight = 30.sp), color = Color(0xFF9AA3B5))
        }
        for (line in lines.takeLast(50)) Text(line, style = style, color = Color.White, modifier = Modifier.padding(bottom = 6.dp))
        if (interim.isNotEmpty()) Text(interim, style = style, color = Interim)
    }
}
