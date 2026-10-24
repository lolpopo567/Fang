package app.fang.ui

import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.onRoot
import com.github.takahirom.roborazzi.captureRoboImage
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import org.robolectric.annotation.GraphicsMode

/**
 * Renders every screen, light and dark, without a phone. Each test also checks that the screen composes.
 * ./gradlew testDebugUnitTest -Pscreenshots writes the images to app/build/screenshots/.
 */
@RunWith(RobolectricTestRunner::class)
@GraphicsMode(GraphicsMode.Mode.NATIVE)
@Config(sdk = [35], qualifiers = "w400dp-h1300dp-xxhdpi")
class ScreenshotTest {
    @get:Rule val compose = createComposeRule()

    private fun shoot(screen: Screen, theme: String, lang: String = "th") {
        compose.setContent { FangTheme(theme, lang) { FangApp(start = screen) } }
        compose.waitForIdle()
        compose.onRoot().captureRoboImage("build/screenshots/$theme-${screen.name.lowercase()}-$lang.png")
    }

    @Test fun homeLight() = shoot(Screen.Home, "light")
    @Test fun homeDark() = shoot(Screen.Home, "dark")
    @Test fun homeEnglish() = shoot(Screen.Home, "light", "en")
    @Test fun captionsLight() = shoot(Screen.Captions, "light")
    @Test fun captionsDark() = shoot(Screen.Captions, "dark")
    @Test fun speakLight() = shoot(Screen.Speak, "light")
    @Test fun speakDark() = shoot(Screen.Speak, "dark")
    @Test fun signLight() = shoot(Screen.Sign, "light")
    @Test fun signDark() = shoot(Screen.Sign, "dark")
    @Test fun settingsLight() = shoot(Screen.Settings, "light")
    @Test fun settingsDark() = shoot(Screen.Settings, "dark")
}
