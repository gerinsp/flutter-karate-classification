import 'dart:async';
import 'dart:convert';
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;

List<CameraDescription>? cameras;

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  cameras = await availableCameras();
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Karate Pose Classifier',
      theme: ThemeData(primarySwatch: Colors.blue),
      home: PoseClassifierPage(),
      debugShowCheckedModeBanner: false,
    );
  }
}

class PoseClassifierPage extends StatefulWidget {
  @override
  _PoseClassifierPageState createState() => _PoseClassifierPageState();
}

class _PoseClassifierPageState extends State<PoseClassifierPage> {
  CameraController? _controller;
  Timer? _timer;
  int _selectedCameraIdx = 0;
  bool _isProcessing = false;

  String _hasilPrediksi = "-";
  double _confidence = 0.0;
  int _frameCount = 0; // ✅ Tambahkan counter frame

  @override
  void initState() {
    super.initState();
    _initializeCamera(_selectedCameraIdx);
  }

  Future<void> _initializeCamera(int cameraIndex) async {
    _controller = CameraController(
      cameras![cameraIndex],
      ResolutionPreset.medium,
      enableAudio: false,
    );

    try {
      await _controller!.initialize();
      if (mounted) setState(() {});
      _startPeriodicCapture();
    } catch (e) {
      print("Error initializing camera: $e");
    }
  }

  void _startPeriodicCapture() {
    _timer?.cancel();
    _timer = Timer.periodic(Duration(milliseconds: 100), (_) async {
      if (!_controller!.value.isInitialized || _isProcessing) return;
      await _captureAndSend();
    });
  }

  Future<void> _captureAndSend() async {
    _isProcessing = true;

    try {
      final XFile file = await _controller!.takePicture();
      final bytes = await file.readAsBytes();

      final response = await http.post(
        Uri.parse("http://192.168.1.5:8000/predict"),
        headers: {'Content-Type': 'application/json'},
        body: json.encode({'image': base64Encode(bytes)}),
      ).timeout(Duration(seconds: 3));

      if (response.statusCode == 200) {
        final result = json.decode(response.body);
        if (mounted) {
          setState(() {
            _frameCount++; // ✅ Tambahkan setiap berhasil kirim frame
            _hasilPrediksi = result['label'] ?? "-";
            _confidence = (result['confidence'] ?? 0.0).toDouble();
          });
        }
      } else {
        print("HTTP Error: ${response.statusCode}");
      }
    } catch (e) {
      print("Error: $e");
    }

    _isProcessing = false;
  }

  void _gantiKamera() async {
    _selectedCameraIdx = (_selectedCameraIdx + 1) % cameras!.length;
    await _initializeCamera(_selectedCameraIdx);
  }

  @override
  void dispose() {
    _timer?.cancel();
    _controller?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('Karate Pose Classifier')),
      body: Column(
        children: [
          if (_controller != null && _controller!.value.isInitialized)
            Expanded(
              child: FittedBox(
                fit: BoxFit.cover,
                child: SizedBox(
                  width: _controller!.value.previewSize!.height,
                  height: _controller!.value.previewSize!.width,
                  child: CameraPreview(_controller!),
                ),
              ),
            ),
          Container(
            padding: EdgeInsets.symmetric(vertical: 20),
            child: Column(
              children: [
                Text(
                  "Hasil: $_hasilPrediksi",
                  style: TextStyle(fontSize: 22, fontWeight: FontWeight.bold),
                ),
                Text(
                  "Confidence: ${(_confidence * 100).toStringAsFixed(2)}%",
                  style: TextStyle(fontSize: 16),
                ),
                Text(
                  "Total Frame Dikirim: $_frameCount", // ✅ Tampilkan frame count
                  style: TextStyle(fontSize: 16, color: Colors.grey[700]),
                ),
                SizedBox(height: 10),
                ElevatedButton.icon(
                  onPressed: _gantiKamera,
                  icon: Icon(Icons.cameraswitch),
                  label: Text("Ganti Kamera"),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}
