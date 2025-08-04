import 'dart:typed_data';
import 'dart:ui';
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;
import 'package:image_picker/image_picker.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:flutter/foundation.dart';
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
  bool _isBusy = false;
  bool _isTakingPicture = false;
  int _frameCounter = 0;

  final int _inputSize = 640;
  final double _confThreshold = 0.5;
  final double _iouThreshold = 0.3;

  final ValueNotifier<List<Map<String, dynamic>>> _recognitionsNotifier = ValueNotifier([]);

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
    _recognitionsNotifier.dispose();
    super.dispose();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    final CameraController? cameraController = _cameraController;

    if (cameraController == null || !cameraController.value.isInitialized) return;

    if (state == AppLifecycleState.inactive) {
      cameraController.dispose();
    } else if (state == AppLifecycleState.resumed) {
      _initializeCamera();
    }
  }

  void _initialize() async {
    await _loadModel();
    await _loadLabels();
    await _initializeCamera();
  }

  Future<void> _initializeCamera() async {
    final cameras = await availableCameras();
    final controller = CameraController(
      cameras[0],
      ResolutionPreset.medium,
      enableAudio: false,
      imageFormatGroup: ImageFormatGroup.yuv420,
    );
    _cameraController = controller;

    try {
      await controller.initialize();
      await controller.setFlashMode(FlashMode.off);
      await controller.startImageStream(_processCameraImage);
      if (mounted) {
        setState(() => _isLoaded = true);
      }
    } catch (e) {
      debugPrint("Error initializing camera: $e");
    }
  }

  Future<void> _loadModel() async {
    try {
      _interpreter = await Interpreter.fromAsset('assets/model2.tflite');
    } catch (e) {
      debugPrint("Error loading model: $e");
    }
  }

  Future<void> _loadLabels() async {
    try {
      final labelData = await rootBundle.loadString('assets/labels.txt');
      _labels = labelData.split('\n').where((l) => l.trim().isNotEmpty).toList();
    } catch (e) {
      debugPrint("Error loading labels: $e");
    }
  }

  void _processCameraImage(CameraImage image) {
    _frameCounter++;
    if (_frameCounter % 10 != 0) return;
    if (_isBusy || !_isLoaded) return;

    _isBusy = true;

    Future.microtask(() async {
      final img.Image rgbImage = _convertYUV420toImage(image);
      final recognitions = await compute(_runModelWithoutLoadingModel, {
        'image': rgbImage,
        'labels': _labels,
        'inputSize': _inputSize,
        'confThreshold': _confThreshold,
        'iouThreshold': _iouThreshold,
        'interpreterAddress': _interpreter?.address,
      });
      _recognitionsNotifier.value = recognitions;
      _isBusy = false;
    });
  }

  static Future<List<Map<String, dynamic>>> _runModelWithoutLoadingModel(Map<String, dynamic> data) async {
    final img.Image image = data['image'];
    final List<String> labels = List<String>.from(data['labels']);
    final int inputSize = data['inputSize'];
    final double confThreshold = data['confThreshold'];
    final double iouThreshold = data['iouThreshold'];
    final interpreter = Interpreter.fromAddress(data['interpreterAddress']);

    final resized = img.copyResize(image, width: inputSize, height: inputSize);
    final inputImage = resized.getBytes(order: img.ChannelOrder.rgb).map((e) => e / 255.0).toList();
    final input = Float32List.fromList(inputImage).reshape([1, inputSize, inputSize, 3]);

    final output = List.filled(1 * (labels.length + 4) * 8400, 0.0).reshape([1, labels.length + 4, 8400]);
    interpreter.run(input, output);

    final boxes = <Rect>[];
    final scores = <double>[];
    final classes = <int>[];

    for (int i = 0; i < 8400; i++) {
      double maxScore = 0;
      int classIndex = -1;
      for (int j = 0; j < labels.length; j++) {
        final score = output[0][j + 4][i];
        if (score > maxScore) {
          maxScore = score;
          classIndex = j;
        }
      }
      if (maxScore > confThreshold) {
        final cx = output[0][0][i];
        final cy = output[0][1][i];
        final w = output[0][2][i];
        final h = output[0][3][i];
        final left = (cx - w / 2).clamp(0.0, 1.0);
        final top = (cy - h / 2).clamp(0.0, 1.0);
        final width = w.clamp(0.0, 1.0 - left);
        final height = h.clamp(0.0, 1.0 - top);
        boxes.add(Rect.fromLTWH(left, top, width, height));
        scores.add(maxScore);
        classes.add(classIndex);
      }
    }

    final picked = <int>[];
    final indexes = List.generate(scores.length, (i) => i);
    indexes.sort((a, b) => scores[b].compareTo(scores[a]));

    while (indexes.isNotEmpty) {
      final current = indexes.removeAt(0);
      picked.add(current);
      indexes.removeWhere((i) => _iou(boxes[current], boxes[i]) > iouThreshold);
    }

    return picked.map((i) => {
      "rect": boxes[i],
      "confidence": scores[i],
      "label": labels[classes[i]]
    }).toList();
  }

  static double _iou(Rect a, Rect b) {
    final interX1 = a.left > b.left ? a.left : b.left;
    final interY1 = a.top > b.top ? a.top : b.top;
    final interX2 = (a.right < b.right ? a.right : b.right);
    final interY2 = (a.bottom < b.bottom ? a.bottom : b.bottom);

    final interArea = (interX2 - interX1).clamp(0.0, double.infinity) *
        (interY2 - interY1).clamp(0.0, double.infinity);
    final unionArea = a.width * a.height + b.width * b.height - interArea;
    return (unionArea > 0) ? interArea / unionArea : 0.0;
  }

  img.Image _convertYUV420toImage(CameraImage image) {
    final width = image.width;
    final height = image.height;
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

  Future<void> _takePicture() async {
    if (_isTakingPicture || _cameraController == null) return;
    setState(() => _isTakingPicture = true);

    try {
      final file = await _cameraController!.takePicture();
      if (mounted) {
        await Navigator.push(
          context,
          MaterialPageRoute(builder: (_) => AnalysisScreen(imagePath: file.path)),
        );
      }
    } catch (e) {
      debugPrint("Error taking picture: $e");
    } finally {
      if (mounted) setState(() => _isTakingPicture = false);
    }
  }

  Future<void> _pickImageFromGallery() async {
    if (_isTakingPicture) return;
    final picker = ImagePicker();
    final imgPick = await picker.pickImage(source: ImageSource.gallery);
    if (imgPick != null && mounted) {
      await Navigator.push(
        context,
        MaterialPageRoute(builder: (_) => AnalysisScreen(imagePath: imgPick.path)),
      );
    }
  }

  Future<void> _toggleFlash() async {
    if (_cameraController == null) return;
    try {
      final currentMode = _cameraController!.value.flashMode;
      final nextMode = currentMode == FlashMode.torch ? FlashMode.off : FlashMode.torch;
      await _cameraController!.setFlashMode(nextMode);
      setState(() {});
    } catch (e) {
      debugPrint("Error toggling flash: $e");
    }
  }

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
          ValueListenableBuilder<List<Map<String, dynamic>>>(
            valueListenable: _recognitionsNotifier,
            builder: (context, recognitions, _) {
              final screen = MediaQuery.of(context).size;
              return Stack(
                children: recognitions.map((r) {
                  final rect = r["rect"] as Rect;
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
                          padding: const EdgeInsets.symmetric(horizontal: 4, vertical: 2),
                          color: Colors.amber,
                          child: Text(
                            r['label'],
                            style: const TextStyle(fontSize: 12, color: Colors.black),
                          ),
                        ),
                      ),
                    ),
                  );
                }).toList(),
              );
            },
          ),
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
                  IconButton(
                    icon: const Icon(Icons.photo_library, color: Colors.white),
                    onPressed: _pickImageFromGallery,
                  ),
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
                  IconButton(
                    icon: Icon(
                      _cameraController?.value.flashMode == FlashMode.torch
                          ? Icons.flash_on
                          : Icons.flash_off,
                      color: Colors.white,
                    ),
                    onPressed: _toggleFlash,
                  )
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}