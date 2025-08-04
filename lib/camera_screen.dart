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
  final double _confThreshold = 0.25; // Turunkan dari 0.5 ke 0.25
  final double _iouThreshold = 0.5;   // Naikkan dari 0.3 ke 0.5

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
    try {
      final cameras = await availableCameras();
      if (cameras.isEmpty) {
        debugPrint("No cameras available");
        return;
      }

      final controller = CameraController(
        cameras[0],
        ResolutionPreset.high, // Ubah dari ultraHigh ke high untuk performa
        enableAudio: false,
        imageFormatGroup: ImageFormatGroup.yuv420,
      );
      _cameraController = controller;

      await controller.initialize();
      await controller.setFlashMode(FlashMode.off);
      await controller.startImageStream(_processCameraImage);

      if (mounted) {
        setState(() => _isLoaded = true);
      }
      debugPrint("Camera initialized successfully");
    } catch (e) {
      debugPrint("Error initializing camera: $e");
    }
  }

  Future<void> _loadModel() async {
    try {
      _interpreter = await Interpreter.fromAsset('assets/model2.tflite');
      debugPrint("Model loaded successfully");

      // Debug model info
      final inputTensors = _interpreter!.getInputTensors();
      final outputTensors = _interpreter!.getOutputTensors();
      debugPrint("Input shape: ${inputTensors[0].shape}");
      debugPrint("Output shape: ${outputTensors[0].shape}");
    } catch (e) {
      debugPrint("Error loading model: $e");
    }
  }

  Future<void> _loadLabels() async {
    try {
      final labelData = await rootBundle.loadString('assets/labels.txt');
      _labels = labelData.split('\n')
          .map((l) => l.trim())
          .where((l) => l.isNotEmpty)
          .toList();
      debugPrint("Loaded ${_labels.length} labels: $_labels");
    } catch (e) {
      debugPrint("Error loading labels: $e");
    }
  }

  void _processCameraImage(CameraImage image) {
    _frameCounter++;
    if (_frameCounter % 5 != 0) return; // Proses setiap 5 frame

    if (_isBusy || !_isLoaded || _interpreter == null) return;

    _isBusy = true;

    Future.microtask(() async {
      try {
        final img.Image? rgbImage = _convertYUV420toImage(image);
        if (rgbImage == null) {
          _isBusy = false;
          return;
        }

        final recognitions = await compute(_runModel, {
          'image': rgbImage,
          'labels': _labels,
          'interpreterAddress': _interpreter?.address,
          'inputSize': _inputSize,
          'confThreshold': _confThreshold,
          'iouThreshold': _iouThreshold,
        }).timeout(const Duration(seconds: 3));

        if (recognitions.isNotEmpty) {
          debugPrint("Found ${recognitions.length} detections");
          for (var r in recognitions) {
            debugPrint("Detection: ${r['label']} - Confidence: ${r['confidence'].toStringAsFixed(2)}");
          }
        }

        if (mounted) {
          _recognitionsNotifier.value = recognitions;
        }
      } catch (e) {
        debugPrint("Error in model inference: $e");
      } finally {
        _isBusy = false;
      }
    });
  }

  static Future<List<Map<String, dynamic>>> _runModel(Map<String, dynamic> data) async {
    try {
      final img.Image image = data['image'];
      final List<String> labels = List<String>.from(data['labels']);
      final int inputSize = data['inputSize'];
      final double confThreshold = data['confThreshold'];
      final double iouThreshold = data['iouThreshold'];
      final interpreter = Interpreter.fromAddress(data['interpreterAddress']);

      // Resize gambar ke ukuran input model
      final resized = img.copyResize(image, width: inputSize, height: inputSize);

      // Konversi ke float dan normalisasi ke 0-1
      final pixels = resized.getBytes(order: img.ChannelOrder.rgb);
      final inputImage = Float32List(inputSize * inputSize * 3);

      for (int i = 0; i < pixels.length; i++) {
        inputImage[i] = pixels[i] / 255.0;
      }

      final input = inputImage.reshape([1, inputSize, inputSize, 3]);

      // Siapkan output tensor berdasarkan format YOLO
      final numClasses = labels.length;
      final outputSize = numClasses + 5; // 4 koordinat + 1 objectness + classes
      final numAnchors = 8400; // Sesuaikan dengan model Anda

      final output = List.filled(1 * outputSize * numAnchors, 0.0)
          .reshape([1, outputSize, numAnchors]);

      // Jalankan inference
      interpreter.run(input, output);

      // Parse hasil deteksi
      final detections = <Map<String, dynamic>>[];

      for (int i = 0; i < numAnchors; i++) {
        // Ambil objectness score
        final objectness = output[0][4][i];

        if (objectness > confThreshold) {
          // Cari class dengan confidence tertinggi
          double maxClassScore = 0;
          int bestClass = -1;

          for (int j = 0; j < numClasses; j++) {
            final classScore = output[0][j + 5][i];
            if (classScore > maxClassScore) {
              maxClassScore = classScore;
              bestClass = j;
            }
          }

          // Hitung final confidence
          final confidence = objectness * maxClassScore;

          if (confidence > confThreshold && bestClass >= 0) {
            // Ambil koordinat bounding box
            final cx = output[0][0][i];
            final cy = output[0][1][i];
            final w = output[0][2][i];
            final h = output[0][3][i];

            // Konversi ke format LTWH normalized
            final left = (cx - w / 2).clamp(0.0, 1.0);
            final top = (cy - h / 2).clamp(0.0, 1.0);
            final width = w.clamp(0.0, 1.0 - left);
            final height = h.clamp(0.0, 1.0 - top);

            detections.add({
              "rect": Rect.fromLTWH(left, top, width, height),
              "confidence": confidence,
              "label": labels[bestClass],
              "classIndex": bestClass,
            });
          }
        }
      }

      // Non-Maximum Suppression
      if (detections.isEmpty) return [];

      // Sort berdasarkan confidence
      detections.sort((a, b) => (b["confidence"] as double).compareTo(a["confidence"] as double));

      final picked = <Map<String, dynamic>>[];
      final suppressed = <bool>[];

      for (int i = 0; i < detections.length; i++) {
        suppressed.add(false);
      }

      for (int i = 0; i < detections.length; i++) {
        if (suppressed[i]) continue;

        picked.add(detections[i]);
        final rectA = detections[i]["rect"] as Rect;

        for (int j = i + 1; j < detections.length; j++) {
          if (suppressed[j]) continue;

          final rectB = detections[j]["rect"] as Rect;
          if (_iou(rectA, rectB) > iouThreshold) {
            suppressed[j] = true;
          }
        }
      }

      debugPrint("Detections before NMS: ${detections.length}, after NMS: ${picked.length}");
      return picked;

    } catch (e) {
      debugPrint("Error in _runModel: $e");
      return [];
    }
  }

  static double _iou(Rect a, Rect b) {
    final interLeft = (a.left > b.left) ? a.left : b.left;
    final interTop = (a.top > b.top) ? a.top : b.top;
    final interRight = (a.right < b.right) ? a.right : b.right;
    final interBottom = (a.bottom < b.bottom) ? a.bottom : b.bottom;

    if (interLeft >= interRight || interTop >= interBottom) return 0.0;

    final interArea = (interRight - interLeft) * (interBottom - interTop);
    final areaA = a.width * a.height;
    final areaB = b.width * b.height;
    final unionArea = areaA + areaB - interArea;

    return (unionArea > 0) ? interArea / unionArea : 0.0;
  }

  img.Image? _convertYUV420toImage(CameraImage image) {
    try {
      final width = image.width;
      final height = image.height;

      // Validasi planes
      if (image.planes.length < 3) {
        debugPrint("Invalid image planes: ${image.planes.length}");
        return null;
      }

      final yPlane = image.planes[0];
      final uPlane = image.planes[1];
      final vPlane = image.planes[2];

      final uvRowStride = uPlane.bytesPerRow;
      final uvPixelStride = uPlane.bytesPerPixel ?? 1;

      final rgbImage = img.Image(width: width, height: height);

      for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
          final yIndex = y * yPlane.bytesPerRow + x;
          final uvIndex = (y ~/ 2) * uvRowStride + (x ~/ 2) * uvPixelStride;

          if (yIndex >= yPlane.bytes.length ||
              uvIndex >= uPlane.bytes.length ||
              uvIndex >= vPlane.bytes.length) {
            continue;
          }

          final yValue = yPlane.bytes[yIndex];
          final uValue = uPlane.bytes[uvIndex];
          final vValue = vPlane.bytes[uvIndex];

          // Konversi YUV ke RGB
          final r = (yValue + 1.402 * (vValue - 128)).clamp(0, 255).toInt();
          final g = (yValue - 0.344136 * (uValue - 128) - 0.714136 * (vValue - 128)).clamp(0, 255).toInt();
          final b = (yValue + 1.772 * (uValue - 128)).clamp(0, 255).toInt();

          rgbImage.setPixelRgb(x, y, r, g, b);
        }
      }

      return rgbImage;
    } catch (e) {
      debugPrint("Error converting YUV420 to RGB: $e");
      return null;
    }
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
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text("Error mengambil gambar: $e")),
        );
      }
    } finally {
      if (mounted) setState(() => _isTakingPicture = false);
    }
  }

  Future<void> _pickImageFromGallery() async {
    if (_isTakingPicture) return;

    try {
      final picker = ImagePicker();
      final imgPick = await picker.pickImage(source: ImageSource.gallery);

      if (imgPick != null && mounted) {
        await Navigator.push(
          context,
          MaterialPageRoute(builder: (_) => AnalysisScreen(imagePath: imgPick.path)),
        );
      }
    } catch (e) {
      debugPrint("Error picking image: $e");
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text("Error memilih gambar: $e")),
        );
      }
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
      return const Scaffold(
        body: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              CircularProgressIndicator(),
              SizedBox(height: 16),
              Text("Memuat kamera dan model..."),
            ],
          ),
        ),
      );
    }

    return Scaffold(
      appBar: AppBar(
        title: const Text("Deteksi Kematangan Mangga"),
        backgroundColor: Colors.green,
        foregroundColor: Colors.white,
      ),
      body: Stack(
        fit: StackFit.expand,
        children: [
          // Camera Preview
          CameraPreview(_cameraController!),

          // Bounding Boxes Overlay
          ValueListenableBuilder<List<Map<String, dynamic>>>(
            valueListenable: _recognitionsNotifier,
            builder: (context, recognitions, _) {
              final screen = MediaQuery.of(context).size;
              return Stack(
                children: recognitions.map((recognition) {
                  final rect = recognition["rect"] as Rect;
                  final confidence = recognition["confidence"] as double;
                  final label = recognition["label"] as String;

                  // Pilih warna berdasarkan label
                  Color boxColor = Colors.red;
                  if (label.toLowerCase().contains('ripe') || label.toLowerCase().contains('matang')) {
                    boxColor = Colors.green;
                  } else if (label.toLowerCase().contains('unripe') || label.toLowerCase().contains('mentah')) {
                    boxColor = Colors.orange;
                  }

                  return Positioned(
                    left: rect.left * screen.width,
                    top: rect.top * screen.height,
                    width: rect.width * screen.width,
                    height: rect.height * screen.height,
                    child: Container(
                      decoration: BoxDecoration(
                        border: Border.all(color: boxColor, width: 3),
                        borderRadius: BorderRadius.circular(8),
                      ),
                      child: Align(
                        alignment: Alignment.topLeft,
                        child: Container(
                          padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 3),
                          decoration: BoxDecoration(
                            color: boxColor,
                            borderRadius: const BorderRadius.only(
                              topLeft: Radius.circular(5),
                              bottomRight: Radius.circular(5),
                            ),
                          ),
                          child: Text(
                            "$label ${(confidence * 100).toInt()}%",
                            style: const TextStyle(
                              fontSize: 12,
                              color: Colors.white,
                              fontWeight: FontWeight.bold,
                            ),
                          ),
                        ),
                      ),
                    ),
                  );
                }).toList(),
              );
            },
          ),

          // Info overlay jika ada deteksi
          ValueListenableBuilder<List<Map<String, dynamic>>>(
            valueListenable: _recognitionsNotifier,
            builder: (context, recognitions, _) {
              if (recognitions.isEmpty) return const SizedBox.shrink();

              return Positioned(
                top: 100,
                left: 16,
                right: 16,
                child: Container(
                  padding: const EdgeInsets.all(12),
                  decoration: BoxDecoration(
                    color: Colors.black.withOpacity(0.7),
                    borderRadius: BorderRadius.circular(8),
                  ),
                  child: Text(
                    "Terdeteksi ${recognitions.length} mangga",
                    style: const TextStyle(
                      color: Colors.white,
                      fontSize: 16,
                      fontWeight: FontWeight.bold,
                    ),
                    textAlign: TextAlign.center,
                  ),
                ),
              );
            },
          ),

          // Control buttons
          Positioned(
            bottom: 0,
            left: 0,
            right: 0,
            child: Container(
              height: 120,
              decoration: BoxDecoration(
                gradient: LinearGradient(
                  begin: Alignment.topCenter,
                  end: Alignment.bottomCenter,
                  colors: [
                    Colors.transparent,
                    Colors.black.withOpacity(0.8),
                  ],
                ),
              ),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                children: [
                  // Gallery button
                  Container(
                    decoration: BoxDecoration(
                      shape: BoxShape.circle,
                      color: Colors.white.withOpacity(0.2),
                    ),
                    child: IconButton(
                      icon: const Icon(Icons.photo_library, color: Colors.white, size: 28),
                      onPressed: _pickImageFromGallery,
                    ),
                  ),

                  // Capture button
                  GestureDetector(
                    onTap: _takePicture,
                    child: Container(
                      width: 80,
                      height: 80,
                      decoration: BoxDecoration(
                        shape: BoxShape.circle,
                        border: Border.all(color: Colors.white, width: 5),
                        color: _isTakingPicture
                            ? Colors.grey.withOpacity(0.5)
                            : Colors.transparent,
                      ),
                      child: _isTakingPicture
                          ? const Center(
                        child: CircularProgressIndicator(
                          color: Colors.white,
                          strokeWidth: 2,
                        ),
                      )
                          : null,
                    ),
                  ),

                  // Flash button
                  Container(
                    decoration: BoxDecoration(
                      shape: BoxShape.circle,
                      color: Colors.white.withOpacity(0.2),
                    ),
                    child: IconButton(
                      icon: Icon(
                        _cameraController?.value.flashMode == FlashMode.torch
                            ? Icons.flash_on
                            : Icons.flash_off,
                        color: Colors.white,
                        size: 28,
                      ),
                      onPressed: _toggleFlash,
                    ),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}