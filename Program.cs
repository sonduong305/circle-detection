using System;
using Emgu.CV.Structure;
using System.IO;
using EMU;

namespace Emgu.CV
{
    class Program
    {
        static void Main(string[] args)
        {
            string path = Directory.GetParent(Directory.GetCurrentDirectory()).Parent.FullName.ToString() + "\\error\\";
            int estimatedRadius = 120;
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
                    circle = circleDetector.FindCircle(img, estimatedRadius: estimatedRadius, error: 100);
                    watch.Stop();
                    var elapsedMs = watch.ElapsedMilliseconds;
                    Console.WriteLine("Finished in " + elapsedMs + "ms");
                    img.Draw(circle, new Gray(0), 3);
                    Console.WriteLine("found circle with your estimated radius = " + estimatedRadius);
                }
                catch (Exception e)
                {
                    //circle = new CircleF(new PointF(0, 0), 0);
                    Console.WriteLine("Cannot find any circles with your estimated radius = " + estimatedRadius);
                }
                CvInvoke.Imshow("result", img);
                CvInvoke.WaitKey();
            }
        }
    }
}