// Import library Flutter dan eksternal
import 'dart:typed_data';
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;
import 'package:image_picker/image_picker.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

import 'analysis_screen.dart'; // Layar hasil analisis setelah ambil foto

class CameraScreen extends StatefulWidget {
  const CameraScreen({super.key});

  @override
  State<CameraScreen> createState() => _CameraScreenState();
}

class _CameraScreenState extends State<CameraScreen> with WidgetsBindingObserver {
  // Variabel untuk kamera dan model
  CameraController? _cameraController;
  Interpreter? _interpreter;

  // Label dan hasil deteksi
  List<String> _labels = [];
  List<dynamic> _recognitions = [];

  // Status
  bool _isLoaded = false;
  bool _isBusy = false;
  int _frameCounter = 0;
  bool _isTakingPicture = false;

  // Konfigurasi model YOLO
  final int _inputSize = 640;
  final double _confThreshold = 0.6;
  final double _iouThreshold = 0.5;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _initialize(); // Inisialisasi model, kamera, dan label
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _cameraController?.dispose();
    _interpreter?.close();
    super.dispose();
  }

  // Fungsi utama untuk inisialisasi
  Future<void> _initialize() async {
    await _loadModel();
    await _loadLabels();
    await _initCamera();
    setState(() => _isLoaded = true);
  }

  // Inisialisasi kamera
  Future<void> _initCamera() async {
    final cameras = await availableCameras();
    _cameraController = CameraController(cameras[0], ResolutionPreset.medium, enableAudio: false);
    await _cameraController!.initialize();
    await _cameraController!.setFlashMode(FlashMode.off);
    await _cameraController!.startImageStream(_runInferenceOnStream);
    setState(() {});
  }

  // Load model TFLite
  Future<void> _loadModel() async {
    try {
      _interpreter = await Interpreter.fromAsset('assets/model.tflite');
    } catch (e) {
      debugPrint('Model error: $e');
    }
  }

  // Load label dari file labels.txt
  Future<void> _loadLabels() async {
    try {
      final labelData = await rootBundle.loadString('assets/labels.txt');
      _labels = labelData.split('\n').where((l) => l.isNotEmpty).toList();
    } catch (e) {
      debugPrint('Label error: $e');
    }
  }

  // Fungsi untuk melakukan inferensi pada stream kamera
  void _runInferenceOnStream(CameraImage image) {
    _frameCounter++;
    if (_frameCounter % 30 != 0 || _isBusy) return;

    _isBusy = true;

    Future(() async {
      final img.Image rgbImage = _convertYUV420toImage(image);
      final results = await _runModel(rgbImage);
      if (mounted) {
        setState(() => _recognitions = results);
      }
    }).whenComplete(() => _isBusy = false);
  }

  // Ambil foto dan pindah ke layar analisis
  Future<void> _takePicture() async {
    if (_isTakingPicture || _cameraController == null) return;

    try {
      setState(() => _isTakingPicture = true);
      await _cameraController!.stopImageStream();

      final file = await _cameraController!.takePicture();

      if (mounted) {
        await Navigator.push(
          context,
          MaterialPageRoute(builder: (_) => AnalysisScreen(imagePath: file.path)),
        );
        await _cameraController!.startImageStream(_runInferenceOnStream);
      }
    } catch (e) {
      debugPrint("Take picture error: $e");
    } finally {
      setState(() => _isTakingPicture = false);
    }
  }

  // Proses gambar untuk model
  Future<List<dynamic>> _runModel(img.Image image) async {
    final img.Image resized = img.copyResize(image, width: _inputSize, height: _inputSize);
    final imageBytes = resized.getBytes(order: img.ChannelOrder.rgb).map((e) => e / 255.0).toList();
    final input = Float32List.fromList(imageBytes).reshape([1, _inputSize, _inputSize, 3]);

    final output = List.filled(1 * (_labels.length + 4) * 8400, 0.0).reshape([1, (_labels.length + 4), 8400]);
    _interpreter?.run(input, output);

    final List<List<List<double>>> casted = (output as List)
        .map((e) => (e as List).map((f) => (f as List).map((g) => (g as num).toDouble()).toList()).toList()).toList();

    return _processOutput(casted);
  }

  // Proses hasil model menjadi daftar bounding box
  List<dynamic> _processOutput(List<List<List<double>>> output) {
    List<Rect> boxes = [];
    List<double> scores = [];
    List<int> classes = [];

    for (int i = 0; i < output[0][0].length; i++) {
      double maxScore = 0;
      int maxIndex = -1;

      for (int j = 0; j < _labels.length; j++) {
        final score = output[0][j + 4][i];
        if (score > maxScore) {
          maxScore = score;
          maxIndex = j;
        }
      }

      if (maxScore > _confThreshold) {
        final cx = output[0][0][i];
        final cy = output[0][1][i];
        final w = output[0][2][i];
        final h = output[0][3][i];
        boxes.add(Rect.fromLTWH(cx - w / 2, cy - h / 2, w, h));
        scores.add(maxScore);
        classes.add(maxIndex);
      }
    }

    final selected = _nonMaxSuppression(boxes, scores);

    return selected.map((i) => {
      "rect": boxes[i],
      "confidence": scores[i],
      "label": _labels[classes[i]]
    }).toList();
  }

  // Konversi dari format YUV420 (kamera) ke RGB image
  img.Image _convertYUV420toImage(CameraImage image) {
    final int width = image.width;
    final int height = image.height;
    final uvRowStride = image.planes[1].bytesPerRow;
    final uvPixelStride = image.planes[1].bytesPerPixel!;
    final img.Image imgBuffer = img.Image(width: width, height: height);

    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        final uvIndex = uvPixelStride * (x ~/ 2) + uvRowStride * (y ~/ 2);
        final yp = image.planes[0].bytes[y * width + x];
        final up = image.planes[1].bytes[uvIndex];
        final vp = image.planes[2].bytes[uvIndex];

        int r = (yp + vp * 1.402).round().clamp(0, 255);
        int g = (yp - up * 0.344 - vp * 0.714).round().clamp(0, 255);
        int b = (yp + up * 1.772).round().clamp(0, 255);
        imgBuffer.setPixelRgb(x, y, r, g, b);
      }
    }

    return imgBuffer;
  }

  // Non-max suppression: hilangkan bounding box yang tumpang tindih
  List<int> _nonMaxSuppression(List<Rect> boxes, List<double> scores) {
    List<int> selected = [];
    List<int> indexes = List.generate(scores.length, (i) => i);
    indexes.sort((a, b) => scores[b].compareTo(scores[a]));

    while (indexes.isNotEmpty) {
      int current = indexes.removeAt(0);
      selected.add(current);
      indexes.removeWhere((i) => _iou(boxes[current], boxes[i]) > _iouThreshold);
    }

    return selected;
  }

  // Hitung IoU antara dua box
  double _iou(Rect a, Rect b) {
    final x1 = a.left > b.left ? a.left : b.left;
    final y1 = a.top > b.top ? a.top : b.top;
    final x2 = (a.right < b.right ? a.right : b.right);
    final y2 = (a.bottom < b.bottom ? a.bottom : b.bottom);

    final intersection = (x2 - x1).clamp(0, double.infinity) * (y2 - y1).clamp(0, double.infinity);
    final union = a.width * a.height + b.width * b.height - intersection;
    return intersection / union;
  }

  // Render bounding box ke atas kamera
  List<Widget> _renderBoxes(BuildContext context) {
    final screen = MediaQuery.of(context).size;
    return _recognitions.map((r) {
      final rect = r["rect"] as Rect;
      final label = r["label"] as String;
      final confidence = (r["confidence"] as double).toStringAsFixed(2);
      return Positioned(
        left: rect.left * screen.width,
        top: rect.top * screen.height,
        width: rect.width * screen.width,
        height: rect.height * screen.height,
        child: Container(
          decoration: BoxDecoration(
            border: Border.all(color: Colors.amber, width: 2),
            borderRadius: BorderRadius.circular(6),
          ),
          child: Align(
            alignment: Alignment.topLeft,
            child: Container(
              color: Colors.amber,
              padding: const EdgeInsets.all(4),
              child: Text('$label: $confidence', style: const TextStyle(color: Colors.black)),
            ),
          ),
        ),
      );
    }).toList();
  }

  // Build tampilan utama
  @override
  Widget build(BuildContext context) {
    if (!_isLoaded || _cameraController == null || !_cameraController!.value.isInitialized) {
      return const Scaffold(body: Center(child: CircularProgressIndicator()));
    }

    return Scaffold(
      appBar: AppBar(title: const Text("Deteksi Kematangan Mangga")),
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
              height: 110,
              color: Colors.black.withOpacity(0.5),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                children: [
                  // Buka galeri
                  IconButton(
                    icon: const Icon(Icons.photo_library, color: Colors.white),
                    onPressed: () async {
                      final picker = ImagePicker();
                      final img = await picker.pickImage(source: ImageSource.gallery);
                      if (img != null && mounted) {
                        await Navigator.push(
                          context,
                          MaterialPageRoute(builder: (_) => AnalysisScreen(imagePath: img.path)),
                        );
                      }
                    },
                  ),
                  // Tombol foto
                  GestureDetector(
                    onTap: _takePicture,
                    child: Container(
                      width: 70,
                      height: 70,
                      decoration: BoxDecoration(
                        shape: BoxShape.circle,
                        border: Border.all(color: Colors.white, width: 4),
                      ),
                    ),
                  ),
                  // Flash on/off
                  IconButton(
                    icon: Icon(
                      _cameraController?.value.flashMode == FlashMode.torch ? Icons.flash_on : Icons.flash_off,
                      color: Colors.white,
                    ),
                    onPressed: () async {
                      final mode = _cameraController!.value.flashMode;
                      final newMode = mode == FlashMode.torch ? FlashMode.off : FlashMode.torch;
                      await _cameraController!.setFlashMode(newMode);
                      setState(() {});
                    },
                  )
                ],
              ),
            ),
          )
        ],
      ),
    );
  }
}
