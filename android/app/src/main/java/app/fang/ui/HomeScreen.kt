package app.fang.ui

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.offset
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import app.fang.R

// Bar heights of the hero's sound wave, as in the web app (fractions of the tallest bar).
private val WAVE = listOf(.25f, .2f, .3f, 1f, .2f, .3f, .55f, 1f, .3f, .2f, .15f, .3f, 1f, .6f, .3f, 1f, .3f, .2f, .45f, .6f, .2f)

@Composable
fun HomeScreen(open: (Screen) -> Unit) {
    Column {
        Hero()
        Spacer(Modifier.height(18.dp))
        ModeCard(Tone.Blue, "captions", t("nav.captions"), t("home.captionsDesc")) { open(Screen.Captions) }
        ModeCard(Tone.Orange, "keyboard", t("nav.speak"), t("home.speakDesc")) { open(Screen.Speak) }
        ModeCard(Tone.Violet, "hand", t("nav.sign"), t("home.signDesc"), badge = t("home.beta")) { open(Screen.Sign) }
        ModeCard(Tone.Slate, "settings", t("nav.settings"), t("home.settingsDesc")) { open(Screen.Settings) }
    }
}

@Composable
private fun Hero() {
    val shape = RoundedCornerShape(28.dp)
    Box(
        Modifier
            .padding(top = 8.dp)
            .shadow(16.dp, shape, ambientColor = colors.heroStart, spotColor = colors.heroStart)
            .fillMaxWidth()
            .height(260.dp)
            .clip(shape)
            .background(Brush.linearGradient(listOf(colors.heroStart, colors.heroEnd, Color(0xFF6D74F0)))),
    ) {
        // Soft rings behind the text, like the web hero.
        Canvas(Modifier.matchParentSize()) {
            for ((r, a) in listOf(0.95f to 0.06f, 0.7f to 0.08f)) {
                drawCircle(Color.White.copy(alpha = a), size.height * r, Offset(size.width * 1.0f, size.height * 0.45f))
            }
        }
        Column(Modifier.padding(start = 22.dp, top = 22.dp, end = 22.dp)) {
            Text(t("home.greeting"), color = Color.White.copy(alpha = 0.85f), style = MaterialTheme.typography.bodyLarge)
            Text(t("home.title"), color = Color.White, style = MaterialTheme.typography.headlineMedium)
            Text(t("home.subtitle"), color = Color.White.copy(alpha = 0.9f), style = MaterialTheme.typography.bodyMedium)
        }
        Canvas(
            Modifier
                .align(Alignment.BottomStart)
                .padding(start = 22.dp)
                .size(width = 190.dp, height = 44.dp),
        ) {
            val step = size.width / WAVE.size
            WAVE.forEachIndexed { i, h ->
                drawRoundRect(
                    Color.White.copy(alpha = 0.75f),
                    topLeft = Offset(i * step, size.height * (1 - h)),
                    size = Size(step * 0.55f, size.height * h),
                    cornerRadius = CornerRadius(step * 0.3f),
                )
            }
        }
        Image(
            painterResource(R.drawable.kitten), null,
            Modifier
                .align(Alignment.BottomEnd)
                .offset(x = (-4).dp, y = 14.dp)
                .size(124.dp),
        )
    }
}

@Composable
private fun ModeCard(tone: Tone, iconName: String, title: String, desc: String, badge: String? = null, onClick: () -> Unit) {
    FangCard(
        Modifier
            .fillMaxWidth()
            .padding(vertical = 7.dp),
    ) {
        Row(
            Modifier
                .background(Brush.horizontalGradient(listOf(tone.soft(colors).copy(alpha = 0.55f), colors.surface), endX = 500f))
                .clickable(role = Role.Button, onClick = onClick)
                .padding(18.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            GradientTile(tone, iconName, size = 64.dp)
            Spacer(Modifier.width(18.dp))
            Column(Modifier.weight(1f)) {
                Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                    Text(title, style = MaterialTheme.typography.titleMedium.copy(fontSize = 22.sp))
                    if (badge != null) {
                        Text(
                            badge,
                            style = MaterialTheme.typography.labelSmall,
                            color = colors.violet,
                            modifier = Modifier
                                .clip(RoundedCornerShape(50))
                                .background(colors.violetSoft)
                                .padding(horizontal = 10.dp, vertical = 2.dp),
                        )
                    }
                }
                Text(desc, style = MaterialTheme.typography.bodyMedium, color = colors.muted)
            }
            FangIcon("chevron", colors.text, modifier = Modifier.padding(start = 8.dp))
        }
    }
}
