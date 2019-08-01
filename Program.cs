using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Emgu.CV;
using Emgu.CV.Structure;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using Emgu.CV.Util;
using Emgu.CV.CvEnum;

using AForge.Imaging;
using AForge.Math;
using EMU;

namespace Emgu.CV
{
    class Program
    {

        static void Main(string[] args)
        {


            //string path = "C:\\Users\\Blake\\Desktop\\opencv_thuy\\error\\New folder\\";
            string path = "C:\\Users\\Blake\\Desktop\\opencv_thuy\\error\\";
            string fileName = "DeskCamera2_DeskCamera2_18551.bmp";
            string resultPath = "C:\\result\\";
            int estimatedRadius = 40;
            List<CircleF_contourArea> circles;
            //Bitmap img;
            Image<Gray, Byte> img1,img;

            
            foreach (string file in Directory.EnumerateFiles(path, "*.jpg"))
            {
                string filename = Path.GetFileName(file);
                img1 = new Image<Gray, Byte>(file);

                img = img1.Clone();
                var watch = System.Diagnostics.Stopwatch.StartNew();
                // the code that you want to measure comes here

                circles = FindCircle_contours(img1, estimatedRadius: estimatedRadius);

                watch.Stop();
                var elapsedMs = watch.ElapsedMilliseconds;
                Console.WriteLine("Finished in " + elapsedMs + "ms");
                //ProcessImage(img);
                Console.WriteLine("found " + circles.Count + " circle(s) with your estimated radius = " + estimatedRadius );
                Console.WriteLine("List of circles: ");
                foreach (CircleF_contourArea circle in circles)
                {
                    Console.WriteLine("\t [" + circle.circle.Center + " \t R = " + circle.circle.Radius + " \t score = " + circle.contourArea / circle.circle.Area + " ]");
                    img1.Draw(circle.circle, new Gray(0), 3);
                    


                }
                CvInvoke.Imshow("result", img1);
                CvInvoke.WaitKey();
                img1.Save(resultPath + "fms_Result_" + filename);
                

            }



        }
        static Mat ellipse_kernel_gen(int size)
        {
            Mat kernel = CvInvoke.GetStructuringElement(ElementShape.Ellipse, new Size(size, size), new Point(-1, -1));
            return kernel;
        }
        /// <summary>
        /// Find circles in Image with estimated Radius using contours
        /// </summary>
        /// <returns>
        /// The list of found circles.
        /// </returns>
        /// <param name="img">Source image with Image format.</param>
        /// <param name="estimatedRadius">Expected Radius.</param>
        static List<CircleF_contourArea> FindCircle_contours(Image<Gray, Byte> image, int estimatedRadius)
        {
            
            Image<Gray, Byte> bilateral_filtered_image,edge_detected_image, dilated, eroded,img;
            img = image.Clone();
            bilateral_filtered_image = new Mat().ToImage<Gray,byte>();
            edge_detected_image = new Mat().ToImage<Gray, byte>();
            dilated = img.Clone();
            eroded = new Mat().ToImage<Gray, byte>();
            Mat hierarchy = new Mat();
            CvInvoke.MorphologyEx(img, img, MorphOp.Close, ellipse_kernel_gen(13), new Point(-1, -1), 1, BorderType.Default, new MCvScalar());

            //bilateral_filtered_image = img.SmoothMedian(5);

            CvInvoke.BilateralFilter(img, bilateral_filtered_image, 9, 30, 30);
            CvInvoke.Canny(bilateral_filtered_image, edge_detected_image, 25, 25);

            CvInvoke.MorphologyEx(edge_detected_image, eroded, MorphOp.Close, ellipse_kernel_gen(11), new Point(-1, -1), 1, BorderType.Default, new MCvScalar());
            VectorOfVectorOfPoint contours = new VectorOfVectorOfPoint();
            CvInvoke.FindContours(eroded, contours, hierarchy, RetrType.List, ChainApproxMethod.ChainApproxSimple);
            List<CircleF_contourArea> circles = new List<CircleF_contourArea>();
            
            for (int i = 0; i < contours.Size; i++)
            {
                using (VectorOfPoint contour = contours[i])
                {
                    Rectangle r = CvInvoke.BoundingRectangle(contour);
                    double x, y, w, h;
                    if (isSquare(r.Width, r.Height))
                    {
                        w = r.Width;
                        h = r.Height;

                        double rect_area = ((w * w) / 4) * 3.14;
                        CircleF circle = CvInvoke.MinEnclosingCircle(contour);
                        double circle_area = circle.Radius * circle.Radius * 3.14;
                        //Console.WriteLine("Circle: rect area : " + rect_area.ToString() + ", circle area: " + circle_area.ToString()+ " R = " +circle.Radius.ToString());
                        if ((Math.Abs(rect_area - circle_area) < rect_area / 10) && (Math.Abs(Math.Sqrt(circle_area / 3.14) - estimatedRadius) < 30)) 
                        {
                            CircleF_contourArea temp = new CircleF_contourArea();
                            temp.circle = circle;
                            temp.contourArea = CvInvoke.ContourArea(contour);
                            circles.Add(temp);
                            //Console.WriteLine("Contour area: " + CvInvoke.ContourArea(contour));
                            
                        }
                    }

                }
            }


            //CvInvoke.Imshow("bilateral_filtered_image", bilateral_filtered_image);
            //CvInvoke.Imshow("edge", edge_detected_image);
            //CvInvoke.Imshow("img", img);
            //CvInvoke.Imshow("circle contour", dilated);

            return circles;
        }
        static bool isSquare(int w, int h)
        {
            return (Math.Abs(w - h) < (w / 8)) && (w * h > 20);
        }
        /// <summary>
        /// Find circles in Image with estimated Radius
        /// </summary>
        /// <returns>
        /// The list of found circles.
        /// </returns>
        /// <param name="img">Source image with Image format.</param>
        /// <param name="estimatedRadius">Expected Radius.</param>
        static List<CircleF> FindCircle_hough(Image<Gray, Byte> img, int estimatedRadius)
        {        
            Gray cannyThreshold = new Gray(100);
            Gray cannyThresholdLinking = new Gray(200);
            Gray circleAccumulatorThreshold = new Gray(60);
            img = img.SmoothMedian(13);
            float dp = 0.5f;


            CircleF[] circles = img.HoughCircles(cannyThreshold, circleAccumulatorThreshold, dp, 30, estimatedRadius - 25, estimatedRadius + 15)[0];
            while ((dp < 4) && ((circles.Length == 0) || (circles[0].Radius == 0)))
            {
                dp += 0.3f;
                circles = img.HoughCircles(cannyThreshold, circleAccumulatorThreshold, dp, 30, estimatedRadius - 25, estimatedRadius + 15)[0];
            }
            Console.WriteLine("dp = "+ dp);
            //List<CircleF> result = circles.ToList();
            return circles.ToList<CircleF>();
        }

        static int FindCircle_blob(Image<Gray, byte> img)
        {
            Bitmap image = img.ToBitmap();
            BlobCounterBase bc = new BlobCounter();
            // set filtering options
            bc.FilterBlobs = true;
            bc.MinWidth = 5;
            bc.MinHeight = 5;
            // set ordering options
            bc.ObjectsOrder = ObjectsOrder.Size;
            // process binary image
            bc.ProcessImage(image);
            Blob[] blobs = bc.GetObjectsInformation();

            // extract the biggest blob
            if (blobs.Length > 0)
            {
                bc.ExtractBlobsImage(image, blobs[0], true);
            }
            image.Save("C:\\result\\hi.png", ImageFormat.Png);
            Console.WriteLine("Found " + blobs[0].Area + " blobs");
            return 0;
        }

        static double Circle_score(CircleF_contourArea circle, Image<Gray,byte> image)
        {
            Image<Gray, byte> gradient = image.Clone();
            Image<Gray, byte> gradient_cropped = image.Clone();
            Image<Gray, byte> img = image.Clone();
            CvInvoke.MedianBlur(img, img, 13);
            double score = 0;

            Image<Gray, byte> mask = new Image<Gray, byte>(img.Width, img.Height);
            CvInvoke.Circle(mask, Point.Round(circle.circle.Center), (int) circle.circle.Radius, new MCvScalar(255, 255, 255), -1);
            Image<Gray, byte> mask_white = mask.Clone();
            Image<Gray, byte> mask_small = mask.Clone();
            CvInvoke.Circle(mask_small, Point.Round(circle.circle.Center), (int)circle.circle.Radius + 2, new MCvScalar(255), -1);
            CvInvoke.Circle(mask_small, Point.Round(circle.circle.Center), (int)circle.circle.Radius - 2 , new MCvScalar(0), -1);
            Image<Gray, byte> dest = new Image<Gray, byte>(img.Width, img.Height);
            //-1 is to fill the area
            Mat cropped = new Mat();
            CvInvoke.BitwiseAnd(img, mask, cropped);
            
            CvInvoke.Laplacian(img, gradient, DepthType.Default);
            CvInvoke.BitwiseAnd(gradient, mask_small, gradient_cropped);
            
            CvInvoke.Circle(mask, Point.Round(circle.circle.Center), (int)circle.circle.Radius, new MCvScalar(CvInvoke.Mean(cropped, mask).ToArray()[0]), -1);
            CvInvoke.AbsDiff(mask, cropped, mask);
            CvInvoke.Canny(cropped, cropped, 25, 25);
            score = CvInvoke.Sum(cropped).ToArray()[0] / CvInvoke.Sum(mask_white).ToArray()[0];
            CvInvoke.Imshow("cropped", cropped);
            //CvInvoke.WaitKey(0);
            //Matrix<Int32> labels = new Matrix<Int32>(10, 2);
            //CvInvoke.Kmeans(cropped, 2, labels, new MCvTermCriteria(15), 50, KMeansInitType.PPCenters);


            if (score > 0.5)
            {
                return score;
            }
            else
            {
                return 1 - score;
            }
            
        }

    }

}
