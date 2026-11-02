package app.fang.ui

import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.StrokeJoin
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.graphics.vector.addPathNodes
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp

private val iconCache = HashMap<String, ImageVector>()

/** A Lucide line icon from the web app's set (see IconPaths.kt), drawn in the tint colour. */
fun icon(name: String): ImageVector = iconCache.getOrPut(name) {
    val b = ImageVector.Builder(name, 24.dp, 24.dp, 24f, 24f)
    for ((d, filled) in ICON_PATHS.getValue(name)) {
        b.addPath(
            pathData = addPathNodes(d),
            fill = if (filled) SolidColor(Color.Black) else null,
            stroke = SolidColor(Color.Black),
            strokeLineWidth = 2f,
            strokeLineCap = StrokeCap.Round,
            strokeLineJoin = StrokeJoin.Round,
        )
    }
    b.build()
}

@Composable
fun FangIcon(name: String, tint: Color, modifier: Modifier = Modifier, size: Dp = 24.dp) {
    Icon(icon(name), contentDescription = null, tint = tint, modifier = modifier.size(size))
}

val CardShape = RoundedCornerShape(20.dp)

@Composable
fun FangCard(modifier: Modifier = Modifier, content: @Composable () -> Unit) {
    Box(
        modifier
            .shadow(6.dp, CardShape, ambientColor = Color(0x140E1726), spotColor = Color(0x140E1726))
            .clip(CardShape)
            .background(colors.surface)
            .border(1.dp, colors.border, CardShape),
    ) { content() }
}

/** Square tile filled with the mode's gradient and a white icon. */
@Composable
fun GradientTile(tone: Tone, iconName: String, size: Dp = 56.dp, modifier: Modifier = Modifier) {
    val shape = RoundedCornerShape(size * 0.3f)
    Box(
        modifier
            .shadow(10.dp, shape, ambientColor = tone.g1, spotColor = tone.g1)
            .size(size)
            .background(tone.gradient, shape),
        contentAlignment = Alignment.Center,
    ) { FangIcon(iconName, Color.White, size = size * 0.48f) }
}

@Composable
fun ScreenHead(tone: Tone, iconName: String, title: String, sub: String) {
    Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.padding(vertical = 8.dp)) {
        GradientTile(tone, iconName, size = 60.dp)
        Spacer(Modifier.width(16.dp))
        Column {
            Text(title, style = MaterialTheme.typography.titleLarge)
            Text(sub, style = MaterialTheme.typography.bodyMedium, color = colors.muted)
        }
    }
}

private val ButtonShape = RoundedCornerShape(18.dp)

/** Filled gradient button in the mode's colour. [live] turns it red, for "stop listening". */
@Composable
fun PrimaryButton(
    text: String,
    iconName: String?,
    tone: Tone,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
    enabled: Boolean = true,
    live: Boolean = false,
) {
    val brush = if (live) SolidColor(Live) else tone.gradient
    Row(
        modifier
            .alpha(if (enabled) 1f else 0.5f)
            .shadow(if (enabled) 10.dp else 0.dp, ButtonShape, ambientColor = tone.g1, spotColor = tone.g1)
            .clip(ButtonShape)
            .background(brush)
            .clickable(enabled = enabled, role = Role.Button, onClick = onClick)
            .heightIn(min = 60.dp)
            .padding(horizontal = 18.dp, vertical = 12.dp),
        horizontalArrangement = Arrangement.Center,
        verticalAlignment = Alignment.CenterVertically,
    ) {
        if (iconName != null) {
            FangIcon(iconName, Color.White)
            Spacer(Modifier.width(10.dp))
        }
        Text(text, style = MaterialTheme.typography.labelLarge.copy(fontWeight = FontWeight.Bold), color = Color.White)
    }
}

@Composable
fun OutlineButton(
    text: String,
    iconName: String?,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
    tint: Color = colors.text,
    small: Boolean = false,
    ghost: Boolean = false,
) {
    val shape = RoundedCornerShape(if (small) 12.dp else 18.dp)
    Row(
        modifier
            .clip(shape)
            .background(if (ghost) Color.Transparent else colors.surface)
            .border(BorderStroke(1.dp, if (ghost) Color.Transparent else colors.border), shape)
            .clickable(role = Role.Button, onClick = onClick)
            .heightIn(min = if (small) 44.dp else 56.dp)
            .padding(horizontal = if (small) 12.dp else 16.dp, vertical = 8.dp),
        horizontalArrangement = Arrangement.Center,
        verticalAlignment = Alignment.CenterVertically,
    ) {
        if (iconName != null) {
            FangIcon(iconName, tint, size = if (small) 18.dp else 22.dp)
            Spacer(Modifier.width(8.dp))
        }
        Text(text, style = if (small) MaterialTheme.typography.labelMedium else MaterialTheme.typography.labelLarge, color = tint)
    }
}

@Composable
fun Notice(text: String, modifier: Modifier = Modifier) {
    Row(
        modifier
            .fillMaxWidth()
            .clip(CardShape)
            .background(colors.warnBg)
            .padding(16.dp),
    ) {
        FangIcon("alert", colors.warnIcon, modifier = Modifier.padding(top = 2.dp))
        Spacer(Modifier.width(12.dp))
        Text(text, style = MaterialTheme.typography.bodyMedium, color = colors.warnText)
    }
}

@Composable
fun StatusText(text: String) {
    if (text.isNotEmpty()) {
        Text(text, style = MaterialTheme.typography.bodyMedium, color = colors.muted, modifier = Modifier.padding(vertical = 4.dp))
    }
}

@Composable
fun FinePrint(text: String, iconName: String? = null) {
    Row(Modifier.padding(vertical = 8.dp)) {
        if (iconName != null) {
            FangIcon(iconName, colors.muted, size = 18.dp, modifier = Modifier.padding(top = 2.dp))
            Spacer(Modifier.width(8.dp))
        }
        Text(text, style = MaterialTheme.typography.bodySmall, color = colors.muted)
    }
}

@Composable
fun SectionLabel(text: String) {
    Text(text, style = MaterialTheme.typography.titleSmall, color = colors.muted, modifier = Modifier.padding(top = 12.dp, bottom = 4.dp))
}

/** A labelled dropdown, the Android stand-in for the web app's <select>. */
@Composable
fun SelectField(label: String, options: List<Pair<String, String>>, value: String, onChange: (String) -> Unit, modifier: Modifier = Modifier) {
    var open by remember { mutableStateOf(false) }
    val shape = RoundedCornerShape(14.dp)
    Column(modifier.padding(vertical = 6.dp)) {
        Text(label, style = MaterialTheme.typography.labelMedium, color = colors.muted)
        Spacer(Modifier.size(6.dp))
        Box {
            Row(
                Modifier
                    .fillMaxWidth()
                    .clip(shape)
                    .background(colors.surface2)
                    .border(1.dp, colors.border, shape)
                    .clickable(role = Role.DropdownList) { open = true }
                    .heightIn(min = 52.dp)
                    .padding(horizontal = 14.dp, vertical = 10.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Text(options.firstOrNull { it.first == value }?.second ?: value, style = MaterialTheme.typography.bodyLarge, modifier = Modifier.weight(1f))
                FangIcon("chevron", colors.muted, size = 20.dp, modifier = Modifier.padding(start = 8.dp))
            }
            DropdownMenu(expanded = open, onDismissRequest = { open = false }, containerColor = colors.surface) {
                for ((key, text) in options) {
                    DropdownMenuItem(
                        text = { Text(text, style = MaterialTheme.typography.bodyLarge, fontWeight = if (key == value) FontWeight.Bold else null) },
                        onClick = { open = false; onChange(key) },
                    )
                }
            }
        }
    }
}
