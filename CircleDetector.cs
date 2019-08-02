using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using System;
using System.Collections.Generic;
using System.Drawing;

namespace EMU
{
    class CircleDetector 
    {
        List<CircleFWithScore> circles = new List<CircleFWithScore>();
        public CircleF FindCircle(Image<Gray, Byte> image, int estimatedRadius,int error = 30) 
        {
            circles.Clear();
            Image<Gray, Byte> bilateralFilteredImage, edgeDetectedImage, dilated, eroded, img;
            img = image.Clone();
            bilateralFilteredImage = new Mat().ToImage<Gray, byte>();
            edgeDetectedImage = new Mat().ToImage<Gray, byte>();
            dilated = img.Clone();
            eroded = new Mat().ToImage<Gray, byte>();
            Mat hierarchy = new Mat();
            VectorOfVectorOfPoint contours = new VectorOfVectorOfPoint();

            CvInvoke.MorphologyEx(img, img, MorphOp.Close, GenerateEllipseKernel(13), new Point(-1, -1), 1, BorderType.Default, new MCvScalar());
            CvInvoke.BilateralFilter(img, bilateralFilteredImage, 9, 30, 30);
            CvInvoke.Canny(bilateralFilteredImage, edgeDetectedImage, 25, 25);
            CvInvoke.MorphologyEx(edgeDetectedImage, eroded, MorphOp.Close, GenerateEllipseKernel(11), new Point(-1, -1), 1, BorderType.Default, new MCvScalar());
            CvInvoke.FindContours(eroded, contours, hierarchy, RetrType.List, ChainApproxMethod.ChainApproxSimple);
           
            for (int i = 0; i < contours.Size; i++)
            {
                using (VectorOfPoint contour = contours[i])
                {
                    Rectangle r = CvInvoke.BoundingRectangle(contour);
                    double w, h;
                    if (IsSquare(r.Width, r.Height))
                    {
                        w = r.Width;
                        h = r.Height;

                        double rect_area = ((w * w) / 4) * Math.PI;
                        CircleF circle = CvInvoke.MinEnclosingCircle(contour);
                        double circle_area = circle.Radius * circle.Radius * Math.PI;

                        if ((Math.Abs(rect_area - circle_area) < rect_area / 10) 
                            && (Math.Abs(Math.Sqrt(circle_area / 3.14) - estimatedRadius) < error))
                        {
                            CircleFWithScore temp = new CircleFWithScore(circle, CvInvoke.ContourArea(contour) / circle.Area);
                            circles.Add(temp);
                        }
                    }
                }
            }
            return FindHighestScoreCircle();
        }

        bool IsSquare(int weight, int height)
        {
            return (Math.Abs(weight - height) < (weight / 8)) && (weight * height > 100);
        }

        Mat GenerateEllipseKernel(int size)
        {
            Mat kernel = CvInvoke.GetStructuringElement(ElementShape.Ellipse, new Size(size, size), new Point(-1, -1));
            return kernel;
        }

        CircleF FindHighestScoreCircle()
        {
            if(circles.Count != 0)
            {
                CircleFWithScore circleFWithMaxScore = circles[0];
                foreach (CircleFWithScore circle in circles)
                {
                    if (circle.score > circleFWithMaxScore.score)
                    {
                        circleFWithMaxScore = circle;
                    }
                }
                return circleFWithMaxScore.circle;
            }
            else
            {
                throw new IndexOutOfRangeException();
            }
        }
    }

    struct CircleFWithScore
    {
        public CircleF circle;
        public double score;
        public CircleFWithScore(CircleF circle, double score)
        {
            this.circle = circle;
            this.score = score;
        }
    }
}
