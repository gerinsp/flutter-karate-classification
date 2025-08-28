import 'package:flutter/services.dart';
import 'dart:async';
import 'dart:convert';
import 'dart:typed_data';
import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:http/http.dart' as http;
import 'package:image/image.dart' as img;

// ================== CONFIG ==================
const String kServerUrl = 'http://192.168.121.17:8000/predict';
const Duration kRequestTimeout = Duration(seconds: 2);
const int kSendEveryNthFrame = 3;
// ============================================

List<CameraDescription>? cameras;

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  cameras = await availableCameras();
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Karate Pose Classifier',
      theme: ThemeData(primarySwatch: Colors.blue),
      debugShowCheckedModeBanner: false,
      home: const PoseClassifierPage(),
    );
  }
}

// ================== DATA MODELS ==================
class FeedbackItem {
  final String part;
  final String message;
  final double score;
  final List<int> landmarkIndices;

  FeedbackItem({
    required this.part,
    required this.message,
    required this.score,
    required this.landmarkIndices,
  });

  factory FeedbackItem.fromJson(Map<String, dynamic> j) => FeedbackItem(
        part: j['part'] ?? '',
        message: j['message'] ?? '',
        score: (j['score'] ?? 0.0).toDouble(),
        landmarkIndices:
            (j['landmark_indices'] as List?)?.map((e) => e as int).toList() ??
                const <int>[],
      );
}

class PredictResult {
  final String label;
  final double confidence;
  final List<FeedbackItem> feedback;
  final List<double>? pose2d; // 33 * 2 (x1,y1,x2,y2,...)
  final List<double>? leftHand2d; // 21 * 2
  final List<double>? rightHand2d; // 21 * 2

  PredictResult({
    required this.label,
    required this.confidence,
    required this.feedback,
    this.pose2d,
    this.leftHand2d,
    this.rightHand2d,
  });

  factory PredictResult.fromJson(Map<String, dynamic> j) => PredictResult(
        label: j['label'] ?? '-',
        confidence: (j['confidence'] ?? 0.0).toDouble(),
        feedback: ((j['feedback'] as List?) ?? [])
            .map((e) => FeedbackItem.fromJson(e as Map<String, dynamic>))
            .toList(),
        pose2d: (j['pose_2d'] as List?)
            ?.map((e) => (e as num).toDouble())
            .toList(),
        leftHand2d: (j['left_hand_2d'] as List?)
            ?.map((e) => (e as num).toDouble())
            .toList(),
        rightHand2d: (j['right_hand_2d'] as List?)
            ?.map((e) => (e as num).toDouble())
            .toList(),
      );
}

// ================== PAGE ==================
class PoseClassifierPage extends StatefulWidget {
  const PoseClassifierPage({super.key});

  @override
  State<PoseClassifierPage> createState() => _PoseClassifierPageState();
}

class _PoseClassifierPageState extends State<PoseClassifierPage> {
  CameraController? _controller;
  bool _processing = false;
  bool _showOverlay = true;

  String _label = '-';
  double _confidence = 0.0;
  int _frameCount = 0;

  List<double>? _pose2d;
  Set<int> _badPoseIdx = <int>{};
  List<FeedbackItem> _feedback = <FeedbackItem>[];

  @override
  void initState() {
    super.initState();
    _initCamera();
  }

  Future<void> _initCamera() async {
    try {
      final CameraDescription cam = cameras!.firstWhere(
        (c) => c.lensDirection == CameraLensDirection.back,
        orElse: () => cameras!.first,
      );
      _controller = CameraController(
        cam,
        ResolutionPreset.low,
        imageFormatGroup: ImageFormatGroup.yuv420,
        enableAudio: false,
      );
      await _controller!.initialize();
      await _controller!.setFlashMode(FlashMode.off);

      await _controller!.startImageStream((CameraImage img) {
        if (_processing) return;
        _processing = true;
        _handleFrame(img).whenComplete(() => _processing = false);
      });

      if (mounted) setState(() {});
    } catch (e) {
      if (kDebugMode) {
        // ignore: avoid_print
        print('Camera init error: $e');
      }
    }
  }

  @override
  void dispose() {
    _controller?.dispose();
    super.dispose();
  }

  int _rotationDegrees(DeviceOrientation o, int sensor) {
    switch (o) {
      case DeviceOrientation.portraitUp:
        return sensor % 360;
      case DeviceOrientation.landscapeLeft:
        return (sensor - 90) % 360;
      case DeviceOrientation.portraitDown:
        return (sensor + 180) % 360;
      case DeviceOrientation.landscapeRight:
        return (sensor + 90) % 360;
      default:
        return 0;
    }
  }

  Future<void> _handleFrame(CameraImage img) async {
    try {
      // Throttle pengiriman frame
      if (_frameCount > 0 && _frameCount % kSendEveryNthFrame != 0) {
        _frameCount++;
        return;
      }

      final orientation = _controller!.value.deviceOrientation;
      final sensor = _controller!.description.sensorOrientation; // 0/90/180/270
      final int rotationDeg = _rotationDegrees(orientation, sensor);

      // Rotate di isolate sebelum encode JPEG
      final Uint8List jpegBytes =
          await compute(_convertYUV420ToJpegWithRotation, {
        'image': img,
        'rotation': rotationDeg,
        'mirror': false, // kamera belakang
      });

      _frameCount++;

      final String b64 = base64Encode(jpegBytes);
      final http.Response resp = await http
          .post(
            Uri.parse(kServerUrl),
            headers: {'Content-Type': 'application/json'},
            body: jsonEncode({'image': b64}),
          )
          .timeout(kRequestTimeout);

      if (!mounted) return;

      if (resp.statusCode == 200) {
        final Map<String, dynamic> data = jsonDecode(resp.body);
        final PredictResult r = PredictResult.fromJson(data);

        final Set<int> badIdx = <int>{};
        for (final f in r.feedback) {
          for (final i in f.landmarkIndices) {
            if (i >= 0 && i < 33) badIdx.add(i);
          }
        }

        setState(() {
          _label = r.label;
          _confidence = r.confidence;
          _feedback = r.feedback;
          _pose2d = r.pose2d; // normalized 0..1
          _badPoseIdx = badIdx;
        });
      } else {
        if (kDebugMode) {
          // ignore: avoid_print
          print('HTTP ${resp.statusCode}: ${resp.body}');
        }
      }
    } catch (e) {
      if (kDebugMode) {
        // ignore: avoid_print
        print('Frame error: $e');
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    final controller = _controller;
    return Scaffold(
      appBar: AppBar(
        title: const Text('Karate Pose Classifier'),
        actions: [
          IconButton(
            tooltip: _showOverlay ? 'Hide overlay' : 'Show overlay',
            onPressed: () => setState(() => _showOverlay = !_showOverlay),
            icon:
                Icon(_showOverlay ? Icons.visibility : Icons.visibility_off),
          ),
        ],
      ),
      body: Column(
        children: [
          if (controller != null && controller.value.isInitialized)
            Expanded(
              child: LayoutBuilder(
                builder: (context, constraints) {
                  final previewSize = controller.value.previewSize!;
                  return Stack(
                    fit: StackFit.expand,
                    children: [
                      CameraPreview(controller),
                      if (_showOverlay && _pose2d != null)
                        CustomPaint(
                          painter: PosePainter(
                            normalizedPose: _pose2d!,
                            badIdx: _badPoseIdx,
                            imageSize: Size(
                              previewSize.width,
                              previewSize.height,
                            ),
                            fit: BoxFit.cover, // sama dgn CameraPreview
                            mirrorX: false,    // kamera belakang
                          ),
                        ),
                    ],
                  );
                },
              ),
            )
          else
            const Expanded(
              child: Center(child: CircularProgressIndicator()),
            ),
          Container(
            width: double.infinity,
            padding:
                const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
            decoration: BoxDecoration(
              color: Theme.of(context).colorScheme.surface,
              boxShadow: [
                BoxShadow(
                  blurRadius: 6,
                  spreadRadius: 0,
                  offset: const Offset(0, -2),
                  color: Colors.black.withOpacity(0.05),
                ),
              ],
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  mainAxisAlignment:
                      MainAxisAlignment.spaceBetween,
                  children: [
                    Expanded(
                      child: Text(
                        'Hasil: $_label',
                        style: const TextStyle(
                          fontSize: 20,
                          fontWeight: FontWeight.bold,
                        ),
                        overflow: TextOverflow.ellipsis,
                      ),
                    ),
                    Text('Conf: ${(_confidence * 100).toStringAsFixed(1)}%'),
                  ],
                ),
                const SizedBox(height: 4),
                Text('Frame terkirim: $_frameCount',
                    style: TextStyle(color: Colors.grey.shade700)),
                const SizedBox(height: 8),
                if (_feedback.isNotEmpty) const Text('Feedback:'),
                if (_feedback.isNotEmpty)
                  ..._feedback.map((f) => Padding(
                        padding: const EdgeInsets.only(top: 4),
                        child: Row(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            const Text('• '),
                            Expanded(
                              child: Text(
                                '${f.part}: ${f.message}',
                                style: TextStyle(
                                  color: Colors.red.shade700,
                                ),
                              ),
                            ),
                          ],
                        ),
                      )),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

// ================== PAINTER ==================
class PosePainter extends CustomPainter {
  final List<double> normalizedPose; // length 66 (x1,y1,x2,y2,...)
  final Set<int> badIdx; // indeks 0..32

  final Size imageSize;
  final BoxFit fit;
  final bool mirrorX;

  PosePainter({
    required this.normalizedPose,
    required this.badIdx,
    required this.imageSize,
    this.fit = BoxFit.cover,
    this.mirrorX = false,
  });

  // subset edges mediapipe pose (33 keypoints)
  static const List<List<int>> edges = [
    [11, 12], [11, 13], [13, 15], [12, 14], [14, 16], // arms
    [11, 23], [12, 24], [23, 24], // torso
    [23, 25], [25, 27], [24, 26], [26, 28], // thighs-knees-ankles
    [27, 29], [29, 31], [28, 30], [30, 32], // ankles-feet
  ];

  Rect _destRect(Size viewSize) {
    final fs = applyBoxFit(fit, imageSize, viewSize);
    final dst = fs.destination;
    return Rect.fromLTWH(
      (viewSize.width - dst.width) / 2,
      (viewSize.height - dst.height) / 2,
      dst.width,
      dst.height,
    );
  }

  List<Offset> _toOffsets(Size viewSize) {
    final Rect dst = _destRect(viewSize);
    final double sx = dst.width, sy = dst.height;

    final List<Offset> pts = <Offset>[];
    for (int i = 0; i < 33; i++) {
      double x = normalizedPose[i * 2];     // 0..1
      double y = normalizedPose[i * 2 + 1]; // 0..1

      if (mirrorX) x = 1.0 - x;

      pts.add(Offset(dst.left + x * sx, dst.top + y * sy));
    }
    return pts;
  }

  @override
  void paint(Canvas canvas, Size size) {
    final List<Offset> pts = _toOffsets(size);

    final Paint lineGood = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2;
    final Paint lineBad = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3;
    final Paint dotGood = Paint()..style = PaintingStyle.fill;
    final Paint dotBad = Paint()..style = PaintingStyle.fill;

    // draw skeleton lines
    for (final e in edges) {
      if (e[0] < pts.length && e[1] < pts.length) {
        final bool anyBad = badIdx.contains(e[0]) || badIdx.contains(e[1]);
        canvas.drawLine(pts[e[0]], pts[e[1]], anyBad ? lineBad : lineGood);
      }
    }
    // draw joints
    for (int i = 0; i < pts.length; i++) {
      canvas.drawCircle(pts[i], badIdx.contains(i) ? 4 : 3,
          badIdx.contains(i) ? dotBad : dotGood);
    }
  }

  @override
  bool shouldRepaint(covariant PosePainter old) {
    return old.normalizedPose != normalizedPose ||
        old.badIdx != badIdx ||
        old.imageSize != imageSize ||
        old.mirrorX != mirrorX ||
        old.fit != fit;
  }
}

// ================== IMAGE CONVERSION (ISOLATE) ==================
Uint8List _convertYUV420ToJpegWithRotation(Map args) {
  final CameraImage image = args['image'] as CameraImage;
  final int rotation = (args['rotation'] as int?)?.abs() ?? 0;
  final bool mirror = (args['mirror'] as bool?) ?? false;

  final Uint8List rgb = _yuv420ToRgb(image);
  img.Image base = img.Image.fromBytes(
    width: image.width,
    height: image.height,
    bytes: rgb.buffer,
    numChannels: 3,
  );

  if (rotation != 0) {
    base = img.copyRotate(base, angle: rotation % 360);
  }
  if (mirror) {
    base = img.flipHorizontal(base);
  }
  return Uint8List.fromList(img.encodeJpg(base, quality: 80));
}

// Konversi CameraImage (YUV420) -> RGB bytes
Uint8List _yuv420ToRgb(CameraImage image) {
  final int width = image.width;
  final int height = image.height;

  final int yRowStride = image.planes[0].bytesPerRow;
  final int uvRowStride = image.planes[1].bytesPerRow;
  final int uvPixelStride = image.planes[1].bytesPerPixel ?? 1;

  final Uint8List yBuffer = image.planes[0].bytes;
  final Uint8List uBuffer = image.planes[1].bytes;
  final Uint8List vBuffer = image.planes[2].bytes;

  final Uint8List rgb = Uint8List(width * height * 3);
  int idx = 0;

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      final int yp = y * yRowStride + x;
      final int uvIndex = uvPixelStride * (x >> 1) + uvRowStride * (y >> 1);

      final int yVal = yBuffer[yp];
      final int uVal = uBuffer[uvIndex];
      final int vVal = vBuffer[uvIndex];

      final int r = (yVal + 1.370705 * (vVal - 128)).clamp(0, 255).toInt();
      final int g = (yVal - 0.337633 * (uVal - 128) - 0.698001 * (vVal - 128))
          .clamp(0, 255)
          .toInt();
      final int b = (yVal + 1.732446 * (uVal - 128)).clamp(0, 255).toInt();

      rgb[idx++] = r;
      rgb[idx++] = g;
      rgb[idx++] = b;
    }
  }
  return rgb;
}
