import 'dart:io';
import 'dart:typed_data';
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';
import 'analysis_screen.dart';

class CameraScreen extends StatefulWidget {
  const CameraScreen({super.key});

  @override
  State<CameraScreen> createState() => _CameraScreenState();
}

class _CameraScreenState extends State<CameraScreen> with WidgetsBindingObserver {
  CameraController? _cameraController;
  Interpreter? _interpreter;
  List<String> _labels = [];
  bool _isLoaded = false;
  List<dynamic> _recognitions = [];
  bool _isBusy = false;
  int _frameCounter = 0;
  bool _isTakingPicture = false;

  final int _modelInputSize = 640;
  final double _confidenceThreshold = 0.5;
  final double _iouThreshold = 0.2;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _initialize();
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _cameraController?.dispose();
    _interpreter?.close();
    super.dispose();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    final CameraController? cameraController = _cameraController;

    if (cameraController == null || !cameraController.value.isInitialized) {
      return;
    }

    if (state == AppLifecycleState.inactive) {
      // Saat aplikasi tidak aktif (misal: pindah ke halaman lain),
      // lepaskan controller untuk membebaskan kamera.
      cameraController.dispose();
    } else if (state == AppLifecycleState.resumed) {
      // Saat aplikasi kembali aktif, buat ulang controller kamera.
      _initializeCamera();
    }
  }

  void _initialize() async {
    await _loadModel();
    await _loadLabels();
    await _initializeCamera();
    if (mounted) setState(() => _isLoaded = true);
  }

  Future<void> _initializeCamera() async {
    final cameras = await availableCameras();
    // Gunakan controller baru untuk re-inisialisasi
    final controller = CameraController(cameras[0], ResolutionPreset.medium, enableAudio: false);

    // Ganti controller lama dengan yang baru
    _cameraController = controller;

    try {
      await controller.initialize();
      await controller.setFlashMode(FlashMode.off);
      await controller.startImageStream(_runInferenceOnStream);
    } catch (e) {
      debugPrint("Error initializing camera: $e");
    }

    if (mounted) {
      setState(() {});
    }
  }

  Future<void> _loadModel() async {
    try {
      // Versi stabil tanpa GPU Delegate
      _interpreter = await Interpreter.fromAsset('assets/model.tflite');
    } catch (e) {
      debugPrint("Error loading model: $e");
    }
  }

  Future<void> _loadLabels() async {
    try {
      final labelsData = await rootBundle.loadString('assets/labels.txt');
      _labels = labelsData.split('\n').where((l) => l.isNotEmpty).toList();
    } catch (e) {
      debugPrint("Error loading labels: $e");
    }
  }

  void _runInferenceOnStream(CameraImage cameraImage) {
    _frameCounter++;
    if (_frameCounter % 10 != 0) return; // Frekuensi deteksi lebih rendah untuk stabilitas
    if (_isBusy) return;
    _isBusy = true;

    Future<void>(() async {
      final rgbImage = _convertYUV420toImage(cameraImage);
      final recognitions = await _runInferenceOnImage(rgbImage);
      if (mounted) {
        setState(() => _recognitions = recognitions);
      }
    }).whenComplete(() => _isBusy = false);
  }

  Future<void> _takePicture() async {
    if (_isTakingPicture) return;

    if (_cameraController == null || !_cameraController!.value.isInitialized) {
      debugPrint("Error: Camera not initialized.");
      return;
    }

    try {
      setState(() {
        _isTakingPicture = true;
      });

      // Tidak perlu stop stream jika controller akan di-dispose oleh lifecycle
      final imageFile = await _cameraController!.takePicture();

      if (mounted) {
        await Navigator.push(
          context,
          MaterialPageRoute(
            builder: (context) => AnalysisScreen(imagePath: imageFile.path),
          ),
        );
      }
    } catch (e) {
      debugPrint("Error taking picture: $e");
    } finally {
      if (mounted) {
        setState(() {
          _isTakingPicture = false;
        });
      }
    }
  }

  Future<List<dynamic>> _runInferenceOnImage(img.Image image) async {
    img.Image resizedImage = img.copyResize(image, width: _modelInputSize, height: _modelInputSize);

    var imageBytes = resizedImage.getBytes(order: img.ChannelOrder.rgb);
    var imageAsList = imageBytes.map((b) => b / 255.0).toList();
    var input = Float32List.fromList(imageAsList).reshape([1, _modelInputSize, _modelInputSize, 3]);

    var output = List.filled(1 * (_labels.length + 4) * 8400, 0.0).reshape([1, (_labels.length + 4), 8400]);

    _interpreter?.run(input, output);

    final List<List<List<double>>> doubleOutput = (output as List).map((e) =>
        (e as List).map((f) =>
            (f as List).map((g) => (g as num).toDouble()).toList()
        ).toList()
    ).toList();

    return _processOutput(doubleOutput);
  }

  List<dynamic> _processOutput(List<List<List<double>>> output) {
    List<Rect> boxes = [];
    List<double> scores = [];
    List<int> classIndexes = [];

    for (int i = 0; i < output[0][0].length; i++) {
      double maxScore = 0;
      int maxClassIndex = -1;
      for (int j = 0; j < _labels.length; j++) {
        double score = output[0][j + 4][i];
        if (score > maxScore) {
          maxScore = score;
          maxClassIndex = j;
        }
      }
      if (maxScore > _confidenceThreshold) {
        double cx = output[0][0][i];
        double cy = output[0][1][i];
        double w = output[0][2][i];
        double h = output[0][3][i];
        boxes.add(Rect.fromLTWH(cx - w / 2, cy - h / 2, w, h));
        scores.add(maxScore);
        classIndexes.add(maxClassIndex);
      }
    }
    List<int> nmsIndexes = _nonMaxSuppression(boxes, scores);
    return nmsIndexes.map((index) => {
      "rect": boxes[index],
      "confidence": scores[index],
      "label": _labels[classIndexes[index]],
    }).toList();
  }

  @override
  Widget build(BuildContext context) {
    if (!_isLoaded || _cameraController == null || !_cameraController!.value.isInitialized) {
      return const Scaffold(body: Center(child: CircularProgressIndicator()));
    }
    return Scaffold(
      appBar: AppBar(title: const Text("Live Object Detection")),
      body: Stack(
        fit: StackFit.expand,
        children: [
          CameraPreview(_cameraController!),
          ..._renderBoxes(context),
          Positioned(
            bottom: 0,
            left: 0,
            right: 0,
            child: Container(
              height: 120,
              color: Colors.black.withOpacity(0.5),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                children: [
                  IconButton(icon: const Icon(Icons.flash_off, color: Colors.white, size: 30), onPressed: () => _cameraController?.setFlashMode(FlashMode.off)),
                  GestureDetector(
                    onTap: _takePicture,
                    child: Container(
                      width: 70, height: 70,
                      decoration: BoxDecoration(shape: BoxShape.circle, border: Border.all(color: Colors.white, width: 4)),
                    ),
                  ),
                  IconButton(icon: const Icon(Icons.flash_on, color: Colors.white, size: 30), onPressed: () => _cameraController?.setFlashMode(FlashMode.torch)),
                ],
              ),
            ),
          )
        ],
      ),
    );
  }

  img.Image _convertYUV420toImage(CameraImage image) {
    final int width = image.width;
    final int height = image.height;
    final int uvRowStride = image.planes[1].bytesPerRow;
    final int? uvPixelStride = image.planes[1].bytesPerPixel;
    final yPlane = image.planes[0].bytes;
    final uPlane = image.planes[1].bytes;
    final vPlane = image.planes[2].bytes;

    final img.Image convertedImage = img.Image(width: width, height: height);
    if (uvPixelStride == null) return convertedImage;

    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        final int uvIndex = uvPixelStride * (x / 2).floor() + uvRowStride * (y / 2).floor();
        final int index = y * width + x;
        final yp = yPlane[index];
        final up = uPlane[uvIndex];
        final vp = vPlane[uvIndex];
        int r = (yp + vp * 1.402).round().clamp(0, 255);
        int g = (yp - up * 0.344 - vp * 0.714).round().clamp(0, 255);
        int b = (yp + up * 1.772).round().clamp(0, 255);
        convertedImage.setPixelRgb(x, y, r, g, b);
      }
    }
    return convertedImage;
  }

  List<int> _nonMaxSuppression(List<Rect> boxes, List<double> scores) {
    List<int> selectedIndexes = [];
    List<int> indexes = List.generate(scores.length, (i) => i)..sort((a, b) => scores[b].compareTo(scores[a]));
    while (indexes.isNotEmpty) {
      int current = indexes.removeAt(0);
      selectedIndexes.add(current);
      indexes.removeWhere((i) => _calculateIoU(boxes[current], boxes[i]) > _iouThreshold);
    }
    return selectedIndexes;
  }

  double _calculateIoU(Rect rect1, Rect rect2) {
    final double intersectionX = (rect1.left < rect2.left) ? rect2.left : rect1.left;
    final double intersectionY = (rect1.top < rect2.top) ? rect2.top : rect1.top;
    final double intersectionWidth = ((rect1.left + rect1.width) < (rect2.left + rect2.width)) ? (rect1.left + rect1.width) : (rect2.left + rect2.width) - intersectionX;
    final double intersectionHeight = ((rect1.top + rect1.height) < (rect2.top + rect2.height)) ? (rect1.top + rect1.height) : (rect2.top + rect2.height) - intersectionY;

    final double intersectionArea = intersectionWidth * intersectionHeight;
    final double unionArea = rect1.width * rect1.height + rect2.width * rect2.height - intersectionArea;

    if (unionArea <= 0 || intersectionArea <= 0) return 0.0;

    return intersectionArea / unionArea;
  }

  List<Widget> _renderBoxes(BuildContext context) {
    final screen = MediaQuery.of(context).size;
    return _recognitions.map((re) {
      final rect = re["rect"] as Rect;
      return Positioned(
        left: rect.left * screen.width,
        top: rect.top * screen.height,
        width: rect.width * screen.width,
        height: rect.height * screen.height,
        child: Container(
          decoration: BoxDecoration(border: Border.all(color: Colors.amber, width: 2), borderRadius: BorderRadius.circular(8)),
          child: Align(
            alignment: Alignment.topLeft,
            child: Container(
              color: Colors.black.withOpacity(0.5),
              padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
              child: Text(
                "${re['label']} ${(re['confidence'] * 100).toStringAsFixed(0)}%",
                style: const TextStyle(color: Colors.white, fontSize: 14),
              ),
            ),
          ),
        ),
      );
    }).toList();
  }
}