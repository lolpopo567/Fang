package app.fang.sign

import android.Manifest
import android.content.pm.PackageManager
import android.util.Size
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalView
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.semantics.LiveRegionMode
import androidx.compose.ui.semantics.liveRegion
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.LocalLifecycleOwner
import androidx.lifecycle.viewmodel.compose.viewModel
import app.fang.R
import app.fang.settings.Prefs
import app.fang.ui.FangCard
import app.fang.ui.FangIcon
import app.fang.ui.FinePrint
import app.fang.ui.Live
import app.fang.ui.Notice
import app.fang.ui.OutlineButton
import app.fang.ui.PrimaryButton
import app.fang.ui.ScreenHead
import app.fang.ui.SelectField
import app.fang.ui.StatusText
import app.fang.ui.Tone
import app.fang.ui.colors
import app.fang.ui.t
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

@Composable
fun signLanguageOptions() = listOf("tsl" to t("sign.langTsl"), "ase" to t("sign.langAsl"))

@Composable
fun SignScreen(vm: SignViewModel = viewModel()) {
    val context = LocalContext.current
    val prefs = Prefs.get(context)
    var permissionDenied by remember { mutableStateOf(false) }
    val askCamera = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        permissionDenied = !granted
        if (granted) vm.start()
    }
    // Stop the camera when the user leaves this screen so it is not left running.
    DisposableEffect(Unit) { onDispose { if (vm.running) vm.stop() } }
    val view = LocalView.current
    DisposableEffect(vm.running) {
        view.keepScreenOn = vm.running
        onDispose { view.keepScreenOn = false }
    }

    Column {
        ScreenHead(Tone.Violet, "hand", t("nav.sign"), t("home.signDesc"))
        Notice(t("sign.prototype"), Modifier.padding(vertical = 8.dp))
        if (permissionDenied) Notice(t("sign.permission"), Modifier.padding(bottom = 8.dp))
        Stage(vm)
        Result(vm.prediction)
        Row(horizontalArrangement = Arrangement.spacedBy(10.dp), modifier = Modifier.padding(vertical = 4.dp)) {
            PrimaryButton(
                t(if (vm.running) "sign.stop" else "sign.start"), if (vm.running) "stop" else "camera", Tone.Violet,
                onClick = {
                    val granted = ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED
                    if (vm.running || granted) vm.toggle() else askCamera.launch(Manifest.permission.CAMERA)
                },
                modifier = Modifier.weight(1f),
            )
            OutlineButton(t("sign.flip"), "flip", vm::flip, Modifier.weight(1f))
        }
        SelectField(t("sign.language"), signLanguageOptions(), prefs.signLang, {
            prefs.signLang = it
            vm.loadModel(it)
        })
        Stats(vm)
        Recorder(vm)
    }
}

@Composable
private fun Stage(vm: SignViewModel) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val shape = RoundedCornerShape(24.dp)
    val previewView = remember {
        PreviewView(context).apply {
            scaleType = PreviewView.ScaleType.FIT_CENTER
            implementationMode = PreviewView.ImplementationMode.COMPATIBLE
        }
    }

    DisposableEffect(vm.running, vm.front) {
        if (!vm.running) return@DisposableEffect onDispose {}
        val future = ProcessCameraProvider.getInstance(context)
        var provider: ProcessCameraProvider? = null
        future.addListener({
            if (!vm.running) return@addListener
            try {
                val p = future.get().also { provider = it }
                // 640x480 like the web app: plenty for the hand tracker, and light on battery.
                val res = ResolutionSelector.Builder()
                    .setAspectRatioStrategy(AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY)
                    .setResolutionStrategy(ResolutionStrategy(Size(640, 480), ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER))
                    .build()
                val preview = Preview.Builder().setResolutionSelector(res).build()
                preview.setSurfaceProvider(previewView.surfaceProvider)
                val analysis = ImageAnalysis.Builder()
                    .setResolutionSelector(res)
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                    .build()
                analysis.setAnalyzer(vm.executor, vm::analyze)
                var selector = if (vm.front) CameraSelector.DEFAULT_FRONT_CAMERA else CameraSelector.DEFAULT_BACK_CAMERA
                if (!p.hasCamera(selector)) selector = if (vm.front) CameraSelector.DEFAULT_BACK_CAMERA else CameraSelector.DEFAULT_FRONT_CAMERA
                p.unbindAll()
                p.bindToLifecycle(lifecycleOwner, selector, preview, analysis)
            } catch (e: Exception) {
                vm.failed(e)
            }
        }, ContextCompat.getMainExecutor(context))
        onDispose { provider?.unbindAll() }
    }

    Box(
        Modifier
            .padding(vertical = 8.dp)
            .fillMaxWidth()
            .aspectRatio(3f / 4f)
            .clip(shape)
            .background(Brush.radialGradient(listOf(Color(0xFF1A1745), Color(0xFF07070F)))),
        contentAlignment = Alignment.Center,
    ) {
        if (vm.running) {
            AndroidView({ previewView }, Modifier.fillMaxSize())
            vm.overlay?.let { HandOverlay(it, mirror = vm.front) }
        } else {
            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                Image(painterResource(R.drawable.kitten), null, Modifier.size(170.dp))
                Spacer(Modifier.height(12.dp))
                Row(verticalAlignment = Alignment.CenterVertically) {
                    FangIcon("camera", Color(0xFFB8BFD0), size = 22.dp)
                    Spacer(Modifier.width(8.dp))
                    Text(t("sign.cameraOff"), color = Color(0xFFB8BFD0), style = MaterialTheme.typography.bodyLarge)
                }
            }
        }
    }
}

/** Landmarks over the preview: amber for the 'Left' slot, cyan for 'Right', as in the web app. */
@Composable
private fun HandOverlay(frame: HandFrame, mirror: Boolean) {
    Canvas(Modifier.fillMaxSize()) {
        // The preview is fitted (letterboxed) in the box; draw in the same rectangle.
        val aspect = frame.aspect.toFloat()
        val (w, h) = if (size.width / size.height > aspect) size.height * aspect to size.height else size.width to size.width / aspect
        val left = (size.width - w) / 2
        val top = (size.height - h) / 2
        fun at(p: DoubleArray) = Offset(left + (if (mirror) 1 - p[0].toFloat() else p[0].toFloat()) * w, top + p[1].toFloat() * h)
        for (hand in frame.hands) {
            val color = if (hand.handedness == "Left") Color(0xFFFFB000) else Color(0xFF00D1FF)
            for ((a, b) in HAND_CONNECTIONS) {
                drawLine(color, at(hand.landmarks[a]), at(hand.landmarks[b]), strokeWidth = 3.dp.toPx(), cap = StrokeCap.Round)
            }
            for (p in hand.landmarks) drawCircle(Color.White, 3.5.dp.toPx(), at(p))
        }
    }
}

@Composable
private fun Result(p: Prediction?) {
    val (label, meta, uncertain) = describe(p)
    FangCard(Modifier.fillMaxWidth().padding(vertical = 8.dp)) {
        Row {
            Box(Modifier.width(6.dp).height(96.dp).background(Tone.Violet.gradient))
            Column(
                Modifier
                    .background(Brush.horizontalGradient(listOf(colors.violetSoft, colors.surface)))
                    .fillMaxWidth()
                    .padding(horizontal = 20.dp, vertical = 14.dp)
                    .semantics { liveRegion = LiveRegionMode.Polite },
            ) {
                Text(
                    label,
                    style = MaterialTheme.typography.headlineMedium.copy(fontSize = 36.sp, lineHeight = 46.sp),
                    color = if (uncertain) colors.muted else colors.text,
                )
                if (meta.isNotEmpty()) Text(meta, style = MaterialTheme.typography.bodyMedium, color = colors.muted)
            }
        }
    }
}

@Composable
private fun describe(p: Prediction?): Triple<String, String, Boolean> {
    if (p == null) return Triple("—", "", false)
    val msg = mapOf(Status.NO_MODEL to "sign.noModel", Status.NOT_CONFIGURED to "sign.noEndpoint", Status.ERROR to "sign.onlineError")[p.status]
    if (msg != null) return Triple("—", t(msg) + (p.error?.let { " ($it)" } ?: ""), false)
    if (p.label in NO_SIGN_LABELS) return Triple("—", t("sign.noSign"), false)
    val pct = Math.round(p.confidence * 100)
    val meta = "${t("sign.confidence")} $pct%" + if (p.uncertain) " · ${t("sign.uncertain")}" else ""
    return Triple(p.text ?: p.label ?: "—", meta, p.uncertain)
}

@Composable
private fun Stats(vm: SignViewModel) {
    val (key, extra) = vm.pipeline
    val items = listOf(
        t("sign.statPipeline") to t(key) + extra,
        t("sign.statHands") to (vm.hands?.toString() ?: "—"),
        t("sign.statFps") to (vm.fps?.toString() ?: "—"),
        t("sign.statLatency") to (vm.latencyMs?.let { "$it ms" } ?: "—"),
    )
    Column(Modifier.padding(vertical = 8.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
        for (row in items.chunked(2)) {
            Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                for ((name, value) in row) {
                    FangCard(Modifier.weight(1f)) {
                        Column(Modifier.padding(horizontal = 16.dp, vertical = 12.dp)) {
                            Text(name, style = MaterialTheme.typography.bodySmall, color = colors.muted)
                            Text(value, style = MaterialTheme.typography.titleMedium)
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun Recorder(vm: SignViewModel) {
    val context = LocalContext.current
    val prefs = Prefs.get(context)
    var open by rememberSaveable { mutableStateOf(false) }
    var label by rememberSaveable { mutableStateOf("") }
    val export = rememberLauncherForActivityResult(ActivityResultContracts.CreateDocument("application/json")) { uri ->
        if (uri != null) {
            runCatching { context.contentResolver.openOutputStream(uri)?.use(vm.recorder::export) }
                .onSuccess { vm.recStatus = "rec.exported" to "" }
        }
    }
    FangCard(Modifier.fillMaxWidth().padding(vertical = 8.dp)) {
        Column {
            Row(
                Modifier.fillMaxWidth().clickable { open = !open }.padding(16.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                FangIcon("record", Live, size = 22.dp)
                Spacer(Modifier.width(12.dp))
                Text(t("rec.title"), style = MaterialTheme.typography.titleSmall.copy(fontWeight = FontWeight.Bold), modifier = Modifier.weight(1f))
            }
            AnimatedVisibility(open) {
                Column(Modifier.padding(start = 16.dp, end = 16.dp, bottom = 16.dp)) {
                    FinePrint(t("rec.help"))
                    OutlinedTextField(label, { label = it }, label = { Text(t("rec.label")) }, singleLine = true, modifier = Modifier.fillMaxWidth())
                    OutlinedTextField(prefs.signer, { prefs.signer = it }, label = { Text(t("rec.signer")) }, singleLine = true, modifier = Modifier.fillMaxWidth().padding(top = 6.dp))
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp), modifier = Modifier.padding(top = 10.dp)) {
                        OutlineButton(t("rec.record"), "record", { vm.record(label) }, Modifier.weight(1f), small = true)
                        OutlineButton(t("rec.export"), "download", {
                            export.launch("fang-sign-samples-${SimpleDateFormat("yyyy-MM-dd", Locale.ROOT).format(Date())}.json")
                        }, Modifier.weight(1f), small = true)
                        OutlineButton(t("rec.clear"), "trash", vm::clearSamples, Modifier.weight(1f), small = true)
                    }
                    val (key, extra) = vm.recStatus
                    StatusText(
                        when {
                            key.isNotEmpty() -> t(key) + extra
                            vm.sampleCount > 0 -> t("rec.count") + vm.sampleCount
                            else -> ""
                        },
                    )
                }
            }
        }
    }
}
