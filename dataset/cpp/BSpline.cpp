#include <inlanecruising_plugin/smoothing/BSpline.hpp>

namespace inlanecruising_plugin
{
namespace smoothing
{
void BSpline::setPoints(const std::vector<lanelet::BasicPoint2d>& points)
{
  Eigen::MatrixXd matrix_points(2, points.size());
  int row_index = 0;
  for(auto const point : points){
      matrix_points.col(row_index) << point.x(), point.y();
      row_index++;
  }
  spline_ = Eigen::SplineFitting<Spline2d>::Interpolate(matrix_points,3 );
}
lanelet::BasicPoint2d BSpline::operator()(double t) const
{
  Eigen::VectorXd values = spline_(t);
  lanelet::BasicPoint2d pt = {values.x(), values.y()};
  return pt;
}

lanelet::BasicPoint2d  BSpline::first_deriv(double t) const {
  Eigen::Array2Xd v = spline_.derivatives(t, 1);
  lanelet::BasicPoint2d  output = {v(2), v(3)};
  return output;
}

lanelet::BasicPoint2d  BSpline::second_deriv(double t) const {
  Eigen::Array2Xd v = spline_.derivatives(t, 2);
  lanelet::BasicPoint2d  output = {v(4), v(5)};
  return output;
}

};  // namespace smoothing
};  // namespace inlanecruising_plugin