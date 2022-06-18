package site.isnt.imclassf

import android.Manifest
import android.annotation.SuppressLint
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.os.SystemClock
import android.util.Log
import android.util.Size
import android.widget.Toast
import androidx.annotation.UiThread
import androidx.annotation.WorkerThread
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import org.pytorch.IValue
import org.pytorch.LiteModuleLoader
import org.pytorch.Module
import org.pytorch.Tensor
import org.pytorch.torchvision.TensorImageUtils
import site.isnt.imclassf.databinding.ActivityMainBinding
import java.io.File
import java.nio.FloatBuffer
import java.util.*
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {
    private lateinit var viewBinding: ActivityMainBinding
    private lateinit var cameraExecutor : ExecutorService

    private var mAnalyzeImageErrorState = false
    private var mModule : Module? = null
    private lateinit var mInputTensorBuffer: FloatBuffer
    private lateinit var mInputTensor : Tensor

    private var mMovingAvgSum : Long = 0
    private val mMovingAvgQueue : Queue<Long> = LinkedList()


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        viewBinding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(viewBinding.root)

        if (allPermissionsGranted()) {
            startCamera()
        } else {
            ActivityCompat.requestPermissions(
                this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS)
        }

        cameraExecutor = Executors.newSingleThreadExecutor()
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()
            val preview = Preview.Builder()
                .build()
                .also{
                    it.setSurfaceProvider(viewBinding.viewFinder.surfaceProvider)
                }

            val camaraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            // val analyzer = LumaAnalyzer()
            val imageAnalysis = ImageAnalysis.Builder()
                .setTargetResolution(Size(224, 224))
                .build()
                .also{
                    it.setAnalyzer(cameraExecutor){ imgprx ->
                        imgprx.use{ // 여기서 쓰고 나중에 close를 해주기에 close 따로 안써줘도 상관 X
                            val result = analyzeImage(imgprx)
                            if(result != null)
                                runOnUiThread {
                                    applyToUiImageAnalyzeResult(result)
                                }
                        }
                    }
                }

            try{
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(this, camaraSelector, preview, imageAnalysis)
            } catch (exc : Exception){
                Log.e(TAG, "Use case binding failed", exc)
            }

        }, ContextCompat.getMainExecutor(this))
    }

    /*
    class LumaAnalyzer : ImageAnalysis.Analyzer{
        override fun analyze(image: ImageProxy) {
            // 이미지 Proxy에 의해 카메라로 들어온 이미지를 전달
            val buffer = image.planes[0].buffer
            val data = ByteArray(buffer.remaining())
            val pixels = data.map{ it.toInt() and 0xff }
            val luma = pixels.average()

            Log.i(TAG, luma.toString())
            image.close() // 분석을 마쳤고, 다음 값을 넘어줘라
        }
    }
     */

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(
            baseContext, it) == PackageManager.PERMISSION_GRANTED
    }

    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<String>, grantResults:
        IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (allPermissionsGranted()) {
                startCamera()
            } else {
                Toast.makeText(this,
                    "Permissions not granted by the user.",
                    Toast.LENGTH_SHORT).show()
                finish()
            }
        }
    }
    override fun onDestroy() {
        super.onDestroy()
    }

    @UiThread
    fun applyToUiImageAnalyzeResult(result : AnalysisResult){
        mMovingAvgSum += result.moduleForwardDuration
        mMovingAvgQueue.add(result.moduleForwardDuration)

        if(mMovingAvgQueue.size > MOVING_AVG_PERIOD)
            mMovingAvgSum -= mMovingAvgQueue.remove()

        val list = mutableListOf(
            "%3.2f fps".format(1000f/result.analysisDuration),
            "%d ms".format(1000/result.moduleForwardDuration),
            "%.0f ms avg".format(mMovingAvgSum.toFloat() / MOVING_AVG_PERIOD)
        )

        for (a in 0 until TOP_K){
            list.add("${result.topNClassNames[a]} : %.2f".format(result.topNScores[a]))
        }

        viewBinding.resultText.text = list.joinToString("\n")
    }

    @SuppressLint("UnsafeOptInUsageError")
    @WorkerThread // 다른 스레드에서 실행하는 것임
    fun analyzeImage(image:ImageProxy) : AnalysisResult? {
        val rotationDegrees = image.imageInfo.rotationDegrees // 이미지가 얼마나 돌아갔는가
        return try{
            if(mModule == null){
                val moduleAbsolutePath = File(
                    Utils.assetFilePath(this, "qmv3so.ptl")!!
                ).absolutePath

                mModule = LiteModuleLoader.load(moduleAbsolutePath)

                mInputTensorBuffer = Tensor.allocateFloatBuffer(
                    3 * INPUT_TENSOR_HEIGHT * INPUT_TENSOR_WIDTH // 값의 크기
                )

                mInputTensor = Tensor.fromBlob(
                    mInputTensorBuffer,
                    longArrayOf(
                        1, 3, INPUT_TENSOR_HEIGHT.toLong(), INPUT_TENSOR_WIDTH.toLong()
                    )
                )
            }
            val startTime = SystemClock.elapsedRealtime()
            TensorImageUtils.imageYUV420CenterCropToFloatBuffer(
                image.image,
                rotationDegrees,
                INPUT_TENSOR_WIDTH,
                INPUT_TENSOR_HEIGHT,
                TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,
                TensorImageUtils.TORCHVISION_NORM_STD_RGB,
                mInputTensorBuffer, 0
            )
            val moduleForwardStartTime = SystemClock.elapsedRealtime()
            val outputTensor = mModule!!.forward(IValue.from(mInputTensor)).toTensor()
            val moduleForwardDuration = SystemClock.elapsedRealtime() - moduleForwardStartTime
            val scores = outputTensor.dataAsFloatArray
            val ixs = Utils.topK(scores, TOP_K).asList()
            val topKClassNames = Constants.IMAGENET_CLASSES.sliceArray(ixs)
            val topKScores = scores.sliceArray(ixs)
            val analysisDuration = SystemClock.elapsedRealtime() - startTime
            AnalysisResult(topKClassNames, topKScores, moduleForwardDuration, analysisDuration)
        } catch(e : Exception){
            Log.e(TAG, "Error during analysis", e)
            mAnalyzeImageErrorState = true
            runOnUiThread {
                Toast.makeText(this, e.message, Toast.LENGTH_SHORT).show()
            }
            null
        }
    }

    companion object {
        private const val TAG = "CameraXApp"
        private const val REQUEST_CODE_PERMISSIONS = 10
        private val REQUIRED_PERMISSIONS =
            mutableListOf (
                Manifest.permission.CAMERA,
                Manifest.permission.RECORD_AUDIO
            ).apply {
                if (Build.VERSION.SDK_INT <= Build.VERSION_CODES.P) {
                    add(Manifest.permission.WRITE_EXTERNAL_STORAGE)
                }
            }.toTypedArray()
        private const val INPUT_TENSOR_WIDTH = 224
        private const val INPUT_TENSOR_HEIGHT = 224
        private const val MOVING_AVG_PERIOD = 10
        private const val TOP_K = 3
        const val INTENT_MODULE_ASSET_NAME = "INTENT_MODULE_ASSET_NAME"
    }

    class AnalysisResult(
        val topNClassNames: Array<String>, val topNScores: FloatArray,
        val moduleForwardDuration: Long, val analysisDuration: Long
    )

}