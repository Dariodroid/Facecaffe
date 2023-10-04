
using System.Drawing;
using OpenCvSharp;
using System.Diagnostics.Metrics;



using OpenCvSharp.Dnn;
using Point = OpenCvSharp.Point;
using Size = OpenCvSharp.Size;
using static OpenCvSharp.FileStorage;
using System;

// Cargar los modelos de detección y clasificación de rostros
const string faceProto = "models/deploy.prototxt";
const string faceModel = "models/Widerface-RetinaFace.caffemodel";
//const string spoofProto = "models/spoof_deploy.prototxt";
const string spoofModel = "models/SINet2.onnx"; // Aquí cambias el modelo de clasificación por el que quieres usar
Net faceNet = CvDnn.ReadNetFromCaffe(faceProto, faceModel);
Net spoofNet = CvDnn.ReadNetFromOnnx(spoofModel); 

// Cargar el modelo de detección de rostros desde los archivos prototxt y caffemodel
Net net = CvDnn.ReadNetFromCaffe(faceProto, faceModel);

// Crear un objeto VideoCapture que se conecta a la cámara web con el índice 0
VideoCapture cap = new VideoCapture(0);
while (true)
{
    // Leer la imagen de la cámara y guardarla en un objeto Mat
    Mat frame = new Mat();
    bool ret = cap.Read(frame);
    if (ret == false)
    {
        break;
    }
    int height = frame.Height;
    int width = frame.Width;
    Mat frame_resized = new Mat();
    Cv2.Resize(frame, frame_resized, new Size(300, 300));

    // Convertir la imagen a escala de grises para evitar el error de CvDnn.BlobFromImage

    // Crear un blob con la imagen en escala de grises y algunos ajustes
    Mat blob = CvDnn.BlobFromImage(frame_resized, 1, new OpenCvSharp.Size(0, 0), new Scalar(104, 117, 123));

    net.SetInput(blob);
    Mat detections = net.Forward();

    for (int i = 0; i < detections.Size(2); i++)
    {
        float confidence = detections.At<float>(0, 0, i, 2);
        if (confidence > 0.6)
        {
            float[] box = new float[4];
            for (int j = 0; j < 4; j++)
            {
                box[j] = detections.At<float>(0, 0, i, j + 3);
            }
             int x_start = (int)(box[0] * width);
             int y_start = (int)(box[1] * height);
             int x_end = (int)(box[2] * width);
             int y_end = (int)(box[3] * height);
            Cv2.Rectangle(frame, new Point(x_start, y_start), new Point(x_end, y_end), new Scalar(0, 255, 0), 2);
            Cv2.PutText(frame, $"Conf: {confidence.ToString("N2")}", new Point(x_start, y_start - 5), HersheyFonts.HersheySimplex, 1.2, new Scalar(0, 255, 255), 2);
        }
    }
    Cv2.ImShow("Frame", frame);
    if (Cv2.WaitKey(1) == 27)
    {
        break;
    }
}
cap.Release();
Cv2.DestroyAllWindows();





