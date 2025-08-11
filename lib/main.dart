import 'dart:async';
import 'dart:convert';
import 'dart:typed_data';
import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:image/image.dart' as img;

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
  bool _isProcessing = false;
  String _hasilPrediksi = "-";
  double _confidence = 0.0;
  int _frameCount = 0;

  @override
  void initState() {
    super.initState();
    _initializeCamera();
  }

  Future<void> _initializeCamera() async {
    final camera = cameras!.first;
    _controller = CameraController(
      camera,
      ResolutionPreset.low,
      imageFormatGroup: ImageFormatGroup.yuv420,
    );

    try {
      await _controller!.initialize();
      await _controller!.setFlashMode(FlashMode.off);

      _controller!.startImageStream((CameraImage image) {
        if (!_isProcessing) {
          _isProcessing = true;
          _processCameraImage(image).then((_) {
            _isProcessing = false;
          });
        }
      });

      setState(() {});
    } catch (e) {
      print("Error initializing camera: $e");
    }
  }

  Future<void> _processCameraImage(CameraImage image) async {
    try {
      // skip sebagian frame untuk membatasi FPS
//       if (_frameCount > 0 && _frameCount % 5 != 0) {
//         _frameCount++;
//         return;
//       }

      final jpegBytes = await compute(_convertYUV420ToJpeg, image);
      _frameCount++;

      final base64Image = base64Encode(jpegBytes);

      final response = await http
          .post(
            Uri.parse('http://192.168.55.114:8000/predict'),
            headers: {'Content-Type': 'application/json'},
            body: json.encode({'image': base64Image}),
          )
          .timeout(Duration(seconds: 2));

      if (response.statusCode == 200 && mounted) {
        final result = json.decode(response.body);
        setState(() {
          _hasilPrediksi = result['label'] ?? "-";
          _confidence = (result['confidence'] ?? 0.0).toDouble();
        });
      } else {
        print('HTTP Error: ${response.statusCode}');
      }
    } catch (e) {
      print('Error processing image: $e');
    }
  }

  static Uint8List _convertYUV420ToJpeg(CameraImage image) {
    final width = image.width;
    final height = image.height;

    final rgbBytes = _yuv420ToRgb(image);
    final img.Image baseImage = img.Image.fromBytes(
      width: width,
      height: height,
      bytes: rgbBytes.buffer,   // ByteBuffer sesuai API terbaru
      numChannels: 3,
    );

    return Uint8List.fromList(img.encodeJpg(baseImage, quality: 80));
  }

  static Uint8List _yuv420ToRgb(CameraImage image) {
    final int width = image.width;
    final int height = image.height;
    final yRowStride = image.planes[0].bytesPerRow;
    final uvRowStride = image.planes[1].bytesPerRow;
    final uvPixelStride = image.planes[1].bytesPerPixel!;

    final yBuffer = image.planes[0].bytes;
    final uBuffer = image.planes[1].bytes;
    final vBuffer = image.planes[2].bytes;

    final rgbBuffer = Uint8List(width * height * 3);
    int index = 0;

    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        final yp = y * yRowStride + x;
        final uvIndex = uvPixelStride * (x >> 1) + uvRowStride * (y >> 1);

        final yValue = yBuffer[yp];
        final uValue = uBuffer[uvIndex];
        final vValue = vBuffer[uvIndex];

        final r =
            (yValue + 1.370705 * (vValue - 128)).clamp(0, 255).toInt();
        final g = (yValue -
                0.337633 * (uValue - 128) -
                0.698001 * (vValue - 128))
            .clamp(0, 255)
            .toInt();
        final b =
            (yValue + 1.732446 * (uValue - 128)).clamp(0, 255).toInt();

        rgbBuffer[index++] = r;
        rgbBuffer[index++] = g;
        rgbBuffer[index++] = b;
      }
    }
    return rgbBuffer;
  }

  @override
  void dispose() {
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
            Expanded(child: CameraPreview(_controller!)),
          Padding(
            padding: const EdgeInsets.all(16),
            child: Column(
              children: [
                Text(
                  "Hasil: $_hasilPrediksi",
                  style: TextStyle(
                      fontSize: 22, fontWeight: FontWeight.bold),
                ),
                Text(
                  "Confidence: ${(_confidence * 100).toStringAsFixed(2)}%",
                  style: TextStyle(fontSize: 16),
                ),
                Text(
                  "Total Frame Dikirim: $_frameCount",
                  style: TextStyle(
                      fontSize: 16, color: Colors.grey[700]),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}
