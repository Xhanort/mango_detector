import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';

class AnalysisScreen extends StatefulWidget {
  final String imagePath;
  const AnalysisScreen({super.key, required this.imagePath});

  @override
  State<AnalysisScreen> createState() => _AnalysisScreenState();
}

class _AnalysisScreenState extends State<AnalysisScreen> {
  Interpreter? _interpreter;
  List<String> _labels = [];
  List<dynamic> _recognitions = [];
  bool _isLoading = true;
  Size? _imageSize;

  final int _modelInputSize = 640;
  final double _confidenceThreshold = 0.5;
  final double _iouThreshold = 0.3;

  @override
  void initState() {
    super.initState();
    _analyzeImage();
  }

  @override
  void dispose() {
    _interpreter?.close();
    super.dispose();
  }

  Future<void> _analyzeImage() async {
    await _loadModel();
    await _loadLabels();
    await _getImageSize(widget.imagePath);

    final imageBytes = await File(widget.imagePath).readAsBytes();
    final image = img.decodeImage(imageBytes);

    if (image != null) {
      final recognitions = await _runInferenceOnImage(image);
      if(mounted) {
        setState(() => _recognitions = recognitions);
      }
    }
    if(mounted) setState(() => _isLoading = false);
  }

  Future<void> _loadModel() async {
    try {
      // Hapus GPU delegate untuk kompatibilitas maksimal
      _interpreter = await Interpreter.fromAsset('assets/model2.tflite');
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

  Future<void> _getImageSize(String path) async {
    final bytes = await File(path).readAsBytes();
    final decoded = await decodeImageFromList(bytes);
    if(mounted){
      setState(() {
        _imageSize = Size(decoded.width.toDouble(), decoded.height.toDouble());
      });
    }
  }

  Future<List<dynamic>> _runInferenceOnImage(img.Image image) async {
    img.Image resizedImage = img.copyResize(image, width: _modelInputSize, height: _modelInputSize);

    var imageBytes = resizedImage.getBytes(order: img.ChannelOrder.rgb);
    var imageAsList = imageBytes.map((b) => b / 255.0).toList();
    var input = Float32List.fromList(imageAsList).reshape([1, _modelInputSize, _modelInputSize, 3]);

    var output = List.filled(1 * 8 * 8400, 0.0).reshape([1, 8, 8400]);

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

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text("Hasil Analisis")),
      body: _isLoading || _imageSize == null
          ? const Center(child: CircularProgressIndicator())
          : SingleChildScrollView(
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            _buildImageWithBoxes(),
            const Padding(
              padding: EdgeInsets.fromLTRB(16, 24, 16, 8),
              child: Text("Objek Terdeteksi:", style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
            ),
            _buildDetectedObjectList(),
          ],
        ),
      ),
    );
  }

  Widget _buildImageWithBoxes() {
    return FittedBox(
      child: SizedBox(
        width: _imageSize!.width,
        height: _imageSize!.height,
        child: Stack(
          children: [
            Image.file(File(widget.imagePath)),
            ..._recognitions.map((rec) {
              final rect = rec["rect"] as Rect;
              final imageSize = _imageSize!;
              return Positioned(
                left: rect.left * imageSize.width,
                top: rect.top * imageSize.height,
                width: rect.width * imageSize.width,
                height: rect.height * imageSize.height,
                child: Container(
                  decoration: BoxDecoration(border: Border.all(color: Colors.amber, width: 3), borderRadius: BorderRadius.circular(4)),
                  child: Align(
                    alignment: Alignment.topLeft,
                    child: Container(
                      color: Colors.black.withOpacity(0.6),
                      padding: const EdgeInsets.symmetric(horizontal: 4, vertical: 2),
                      child: Text(
                        "${rec['label']} ${(rec['confidence'] * 100).toStringAsFixed(0)}%",
                        style: const TextStyle(color: Colors.white, fontSize: 12, fontWeight: FontWeight.bold),
                      ),
                    ),
                  ),
                ),
              );
            }).toList(),
          ],
        ),
      ),
    );
  }

  Widget _buildDetectedObjectList() {
    if (_recognitions.isEmpty) {
      return const Padding(
        padding: EdgeInsets.symmetric(horizontal: 16),
        child: Text("Tidak ada objek yang terdeteksi."),
      );
    }
    return ListView.builder(
      shrinkWrap: true,
      physics: const NeverScrollableScrollPhysics(),
      itemCount: _recognitions.length,
      itemBuilder: (context, index) {
        final rec = _recognitions[index];
        return Container(
          margin: const EdgeInsets.symmetric(horizontal: 16, vertical: 4),
          decoration: BoxDecoration(
              color: Colors.black.withOpacity(0.05),
              borderRadius: BorderRadius.circular(8)
          ),
          padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
          child: Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text((rec['label'] as String).toUpperCase(), style: const TextStyle(fontWeight: FontWeight.bold)),
              Text("${((rec['confidence'] as double) * 100).toStringAsFixed(1)}%"),
            ],
          ),
        );
      },
    );
  }
}