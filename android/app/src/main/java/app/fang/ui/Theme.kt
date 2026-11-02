package app.fang.ui

import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Typography
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.ReadOnlyComposable
import androidx.compose.runtime.staticCompositionLocalOf
import androidx.compose.runtime.compositionLocalOf
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.Font
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.sp
import app.fang.R

/** The web app's colour tokens (css/app.css), light and dark. */
data class FangColors(
    val dark: Boolean,
    val bg: Color,
    val surface: Color,
    val surface2: Color,
    val text: Color,
    val muted: Color,
    val border: Color,
    val accent: Color,
    val heroStart: Color,
    val heroEnd: Color,
    val blue: Color, val blueSoft: Color,
    val orange: Color, val orangeSoft: Color,
    val violet: Color, val violetSoft: Color,
    val slate: Color, val slateSoft: Color,
    val glow1: Color,
    val glow2: Color,
    val warnBg: Color,
    val warnText: Color,
    val warnIcon: Color,
)

val LightColors = FangColors(
    dark = false,
    bg = Color(0xFFF3F6FB), surface = Color(0xFFFFFFFF), surface2 = Color(0xFFEAF0F8),
    text = Color(0xFF0E1726), muted = Color(0xFF4A5568), border = Color(0xFFD5DDE9),
    accent = Color(0xFF1559D6), heroStart = Color(0xFF1559D6), heroEnd = Color(0xFF4AA3F0),
    blue = Color(0xFF1559D6), blueSoft = Color(0xFFE2ECFF),
    orange = Color(0xFFB34F00), orangeSoft = Color(0xFFFFEEDD),
    violet = Color(0xFF6A35CF), violetSoft = Color(0xFFEFE7FF),
    slate = Color(0xFF475569), slateSoft = Color(0xFFE8EDF4),
    glow1 = Color(0x333D9CF2), glow2 = Color(0x24A06BFF),
    warnBg = Color(0xFFFFF3D6), warnText = Color(0xFF5A4000), warnIcon = Color(0xFFB07800),
)

val DarkColors = FangColors(
    dark = true,
    bg = Color(0xFF0A0F1E), surface = Color(0xFF131A2E), surface2 = Color(0xFF1B2440),
    text = Color(0xFFEEF2FB), muted = Color(0xFFA9B4CA), border = Color(0xFF283252),
    accent = Color(0xFF7FB2FF), heroStart = Color(0xFF1A3F94), heroEnd = Color(0xFF2A76C9),
    blue = Color(0xFF8DBBFF), blueSoft = Color(0xFF1B2B52),
    orange = Color(0xFFFFAD66), orangeSoft = Color(0xFF3A2614),
    violet = Color(0xFFC2A6FF), violetSoft = Color(0xFF2A1F4D),
    slate = Color(0xFFB6C2D6), slateSoft = Color(0xFF222B44),
    glow1 = Color(0x383D9CF2), glow2 = Color(0x2EA06BFF),
    warnBg = Color(0xFF352A10), warnText = Color(0xFFFFE29A), warnIcon = Color(0xFFFFC94A),
)

/** One colour per mode: blue captions, orange speak, violet signs, slate settings. */
enum class Tone(val g1: Color, val g2: Color) {
    Blue(Color(0xFF1452C8), Color(0xFF2F86E6)),
    Orange(Color(0xFFC2410C), Color(0xFFE8590C)),
    Violet(Color(0xFF5B2BC4), Color(0xFF8B5CF6)),
    Slate(Color(0xFF3B4A63), Color(0xFF7286A5));

    val gradient get() = Brush.linearGradient(listOf(g1, g2))

    fun fg(c: FangColors) = when (this) { Blue -> c.blue; Orange -> c.orange; Violet -> c.violet; Slate -> c.slate }
    fun soft(c: FangColors) = when (this) { Blue -> c.blueSoft; Orange -> c.orangeSoft; Violet -> c.violetSoft; Slate -> c.slateSoft }
}

val Live = Color(0xFFE5262D)

/** IBM Plex Sans Thai, bundled like in the web app (SIL OFL 1.1). */
val PlexThai = FontFamily(
    Font(R.font.plex_thai_400, FontWeight.Normal),
    Font(R.font.plex_thai_600, FontWeight.SemiBold),
    Font(R.font.plex_thai_700, FontWeight.Bold),
)

val LocalColors = staticCompositionLocalOf { LightColors }
val LocalLang = compositionLocalOf { "th" }

/** UI text in the chosen language; falls back to Thai, then to the key, like the web app's t(). */
fun tr(lang: String, key: String): String = STRINGS[lang]?.get(key) ?: STRINGS["th"]?.get(key) ?: key

@Composable
@ReadOnlyComposable
fun t(key: String): String = tr(LocalLang.current, key)

val colors: FangColors
    @Composable @ReadOnlyComposable get() = LocalColors.current

@Composable
fun FangTheme(theme: String, lang: String, content: @Composable () -> Unit) {
    val dark = when (theme) {
        "light" -> false
        "dark" -> true
        else -> isSystemInDarkTheme()
    }
    val c = if (dark) DarkColors else LightColors
    val scheme = (if (dark) darkColorScheme() else lightColorScheme()).copy(
        primary = c.accent, background = c.bg, surface = c.surface, onSurface = c.text, onBackground = c.text,
        surfaceVariant = c.surface2, onSurfaceVariant = c.muted, outline = c.border,
    )
    val base = TextStyle(fontFamily = PlexThai, color = c.text)
    val typography = Typography(
        bodyLarge = base.copy(fontSize = 18.sp, lineHeight = 28.sp),
        bodyMedium = base.copy(fontSize = 16.sp, lineHeight = 24.sp),
        bodySmall = base.copy(fontSize = 14.sp, lineHeight = 21.sp),
        labelLarge = base.copy(fontSize = 17.sp, fontWeight = FontWeight.SemiBold),
        labelMedium = base.copy(fontSize = 14.sp, fontWeight = FontWeight.SemiBold),
        labelSmall = base.copy(fontSize = 12.sp, fontWeight = FontWeight.SemiBold),
        titleLarge = base.copy(fontSize = 26.sp, lineHeight = 34.sp, fontWeight = FontWeight.Bold),
        titleMedium = base.copy(fontSize = 20.sp, lineHeight = 28.sp, fontWeight = FontWeight.Bold),
        titleSmall = base.copy(fontSize = 17.sp, lineHeight = 24.sp, fontWeight = FontWeight.SemiBold),
        headlineMedium = base.copy(fontSize = 32.sp, lineHeight = 42.sp, fontWeight = FontWeight.Bold),
    )
    CompositionLocalProvider(LocalColors provides c, LocalLang provides lang) {
        MaterialTheme(colorScheme = scheme, typography = typography, content = content)
    }
}
