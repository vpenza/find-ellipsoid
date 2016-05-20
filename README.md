# find-ellipsoid
Run optimization problem to find out the best ellipsoid that fits a partial point cloud.

##### Note
For simplicity, the generated partial point cloud and the ellipsoid model do not account for any orientation. Thus, only translation and axes size are considered.

## Installation

##### Dependencies
- [YARP](https://github.com/robotology/yarp)
- [iCub](https://github.com/robotology/icub-main)
- [IpOpt](http://wiki.icub.org/wiki/Installing_IPOPT)

## Options
- **opt**: select the optimization problem
    - 0: unconstrained minimization
    - 1: constrained minimization
- **noise**: select the noise level to inject into the partial input point cloud

## Output
Two `.OFF` files are produced containing the input point cloud and a sampling of the solved ellipsoid.

## Example

```sh
find-ellipsoid --opt 1 --noise 0.002
```
