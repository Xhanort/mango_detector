import 'dart:io';
import 'package:flutter/material.dart';

class ResultScreen extends StatelessWidget {
  final String imagePath;
  final String ripenessStatus;

  const ResultScreen({
    super.key,
    required this.imagePath,
    required this.ripenessStatus,
  });

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text("Hasil Analisis Kematangan"),
        backgroundColor: Colors.green,
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            // Menampilkan gambar yang diambil
            Image.file(
              File(imagePath),
              width: MediaQuery.of(context).size.width * 0.8,
              fit: BoxFit.cover,
            ),
            const SizedBox(height: 24),
            // Menampilkan hasil teks
            Text(
              "Status Kematangan:",
              style: Theme.of(context).textTheme.headlineSmall,
            ),
            const SizedBox(height: 8),
            Text(
              ripenessStatus,
              style: Theme.of(context).textTheme.headlineMedium?.copyWith(
                    fontWeight: FontWeight.bold,
                    color: ripenessStatus.toLowerCase() == 'matang'
                        ? Colors.green
                        : Colors.orange,
              ),
            ),
          ],
        ),
      ),
    );
  }
}