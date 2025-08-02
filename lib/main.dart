import 'package:flutter/material.dart';
import 'camera_screen.dart'; // Arahkan ke file ini

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Object Detector',
      theme: ThemeData(primarySwatch: Colors.blue),
      debugShowCheckedModeBanner: false,
      home: const CameraScreen(), // Mulai dari sini
    );
  }
}