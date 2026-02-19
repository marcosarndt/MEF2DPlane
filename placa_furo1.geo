// Gmsh project created on Thu Feb 12 15:17:04 2026
SetFactory("OpenCASCADE");
//+
Point(1) = {0, 0, 0, 1.0};
//+
Point(2) = {10, 0, 0, 1.0};
//+
Point(3) = {0, 10, 0, 1.0};
//+
Point(4) = {40, 0, 0, 1.0};
//+
Point(5) = {40, 40, 0, 1.0};
//+
Point(6) = {0, 40, 0, 1.0};
//+
Circle(1) = {3, 1, 2};
//+
Line(2) = {2, 4};
//+
Line(3) = {4, 5};
//+
Line(4) = {5, 6};
//+
Line(5) = {6, 3};
//+
Curve Loop(1) = {1, 2, 3, 4, 5};
//+
Surface(1) = {1};
//+
Physical Surface("material1", 6) = {1};
//+
Physical Curve("fixox", 7) = {5};
//+
Physical Curve("fixoy", 8) = {2};
//+
Physical Curve("carga1", 9) = {3};
