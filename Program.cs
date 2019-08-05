using System;
using Emgu.CV.Structure;
using System.IO;
using EMU;
using System.Drawing;

namespace Emgu.CV
{
    class Program
    {
        static void Main(string[] args)
        {
            string path = Directory.GetParent(Directory.GetCurrentDirectory()).Parent.FullName.ToString() + "\\error\\New folder\\";
            int estimatedRadius = 130;
            CircleF circle;
            Image<Gray, Byte> img;
            CircleDetector circleDetector = new CircleDetector();

            foreach (string file in Directory.EnumerateFiles(path, "*.bmp"))
            {
                string fileName = Path.GetFileName(file);
                img = new Image<Gray, Byte>(file);

                try
                {
                    var watch = System.Diagnostics.Stopwatch.StartNew();
                    circle = circleDetector.FindCircle(img, estimatedRadius: estimatedRadius,patternType: 1, error: 20);
                    watch.Stop();
                    var elapsedMs = watch.ElapsedMilliseconds;
                    Console.WriteLine("\nFinished in " + elapsedMs + "ms");
                    img.Draw(circle, new Gray(0), 3);
                    Console.WriteLine("found circle with your estimated radius = " + estimatedRadius);
                    Console.WriteLine("[ " + circle.Center.ToString() + " , R = " + circle.Radius.ToString() + " ]");

                }
                catch (Exception e)
                {
                    //circle = new CircleF(new PointF(0, 0), 0);
                    Console.WriteLine("\nCannot find any circles with your estimated radius = " + estimatedRadius);
                }
                CvInvoke.Imshow("result", img);
                CvInvoke.WaitKey();
            }
        }
    }
}