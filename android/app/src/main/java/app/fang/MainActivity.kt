package app.fang

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import app.fang.settings.Prefs
import app.fang.ui.FangApp
import app.fang.ui.FangTheme

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        val prefs = Prefs.get(this)
        setContent {
            FangTheme(prefs.theme, prefs.uiLang) {
                FangApp()
            }
        }
    }
}
