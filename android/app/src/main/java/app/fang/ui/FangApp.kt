package app.fang.ui

import androidx.activity.compose.BackHandler
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.drawBehind
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.selected
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.withStyle
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.viewmodel.compose.viewModel
import app.fang.R
import app.fang.captions.CaptionsScreen
import app.fang.captions.CaptionsViewModel
import app.fang.settings.SettingsScreen
import app.fang.sign.SignScreen
import app.fang.sign.SignViewModel
import app.fang.speak.BigText
import app.fang.speak.SpeakScreen
import app.fang.speak.SpeakViewModel

enum class Screen { Home, Captions, Speak, Sign, Settings }

private data class Tab(val screen: Screen, val icon: String, val label: String, val tone: Tone?)

/** The whole app: top bar, the current screen, and the floating tab bar, as in the web app. */
@Composable
fun FangApp(start: Screen = Screen.Home) {
    var screen by rememberSaveable { mutableStateOf(start) }
    val captions: CaptionsViewModel = viewModel()
    val speak: SpeakViewModel = viewModel()
    val sign: SignViewModel = viewModel()
    val lang = LocalLang.current
    // The phone's back button returns to the home screen.
    BackHandler(enabled = screen != Screen.Home) { screen = Screen.Home }

    val glow1 = colors.glow1
    val glow2 = colors.glow2
    Box(
        Modifier
            .fillMaxSize()
            .background(colors.bg)
            .drawBehind {
                drawCircle(Brush.radialGradient(listOf(glow1, Color.Transparent), Offset(size.width * 1.1f, 0f), size.width * 0.9f), size.width * 0.9f, Offset(size.width * 1.1f, 0f))
                drawCircle(Brush.radialGradient(listOf(glow2, Color.Transparent), Offset(-size.width * 0.2f, size.height * 0.1f), size.width * 0.7f), size.width * 0.7f, Offset(-size.width * 0.2f, size.height * 0.1f))
            },
    ) {
        val scroll = rememberScrollState()
        LaunchedEffect(screen) { scroll.scrollTo(0) }
        Column(
            Modifier
                .fillMaxSize()
                .statusBarsPadding()
                .imePadding()
                .verticalScroll(scroll),
        ) {
            TopBar(screen == Screen.Settings) { screen = Screen.Settings }
            Column(Modifier.padding(horizontal = 16.dp)) {
                when (screen) {
                    Screen.Home -> HomeScreen { screen = it }
                    Screen.Captions -> CaptionsScreen(captions)
                    Screen.Speak -> SpeakScreen(speak)
                    Screen.Sign -> SignScreen(sign)
                    Screen.Settings -> SettingsScreen(
                        onSpeechChanged = captions::settingsChanged,
                        onSignLangChanged = sign::loadModel,
                        onSignChanged = sign::settingsChanged,
                        onTestVoice = { speak.testVoice(lang) },
                    )
                }
            }
            Spacer(Modifier.navigationBarsPadding().height(120.dp))
        }
        TabBar(screen, Modifier.align(Alignment.BottomCenter)) { screen = it }
        speak.big?.takeIf { screen != Screen.Speak }?.let { BigText(it) { speak.big = null } }
    }
}

@Composable
private fun TopBar(settingsOpen: Boolean, openSettings: () -> Unit) {
    Row(
        Modifier
            .fillMaxWidth()
            .padding(horizontal = 16.dp, vertical = 10.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Image(painterResource(R.drawable.app_logo), null, Modifier.size(44.dp).clip(RoundedCornerShape(12.dp)))
        Spacer(Modifier.width(12.dp))
        Text(
            buildAnnotatedString {
                append("Fang ")
                withStyle(SpanStyle(color = colors.violet)) { append("ฟัง") }
            },
            style = MaterialTheme.typography.titleLarge.copy(fontSize = 28.sp),
            modifier = Modifier.weight(1f),
        )
        val shape = RoundedCornerShape(16.dp)
        Box(
            Modifier
                .shadow(4.dp, shape)
                .size(56.dp)
                .clip(shape)
                .background(if (settingsOpen) colors.slateSoft else colors.surface)
                .border(1.dp, colors.border, shape)
                .clickable(role = Role.Button, onClick = openSettings)
                .semantics { contentDescription = "settings" },
            contentAlignment = Alignment.Center,
        ) {
            FangIcon("settings", colors.text, size = 28.dp)
        }
    }
}

@Composable
private fun TabBar(current: Screen, modifier: Modifier, onSelect: (Screen) -> Unit) {
    val tabs = listOf(
        Tab(Screen.Home, "home", t("nav.home"), null),
        Tab(Screen.Captions, "captions", t("nav.captions"), Tone.Blue),
        Tab(Screen.Speak, "keyboard", t("nav.speak"), Tone.Orange),
        Tab(Screen.Sign, "hand", t("nav.sign"), Tone.Violet),
    )
    val shape = RoundedCornerShape(30.dp)
    Row(
        modifier
            .navigationBarsPadding()
            .padding(horizontal = 12.dp, vertical = 10.dp)
            .shadow(18.dp, shape)
            .clip(shape)
            .background(colors.surface.copy(alpha = 0.96f))
            .border(1.dp, colors.border, shape)
            .padding(horizontal = 6.dp, vertical = 8.dp)
            .fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceEvenly,
    ) {
        for (tab in tabs) {
            val selected = tab.screen == current
            val tone = tab.tone ?: Tone.Blue
            val fg = if (tab.tone == null) colors.accent else tone.fg(colors)
            Column(
                Modifier
                    .weight(1f)
                    .clip(RoundedCornerShape(18.dp))
                    .clickable(role = Role.Tab) { onSelect(tab.screen) }
                    .semantics { this.selected = selected }
                    .padding(vertical = 4.dp),
                horizontalAlignment = Alignment.CenterHorizontally,
            ) {
                Box(
                    Modifier
                        .clip(RoundedCornerShape(50))
                        .background(if (selected) tone.gradient else Brush.linearGradient(listOf(Color.Transparent, Color.Transparent)))
                        .padding(horizontal = 18.dp, vertical = 5.dp),
                ) {
                    FangIcon(tab.icon, if (selected) Color.White else colors.muted, size = 26.dp)
                }
                Text(
                    tab.label,
                    style = MaterialTheme.typography.labelSmall,
                    color = if (selected) fg else colors.muted,
                    maxLines = 1,
                    modifier = Modifier.padding(top = 3.dp),
                )
            }
        }
    }
}
