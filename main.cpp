/*
* Copyright (C) 2016 iCub Facility - Istituto Italiano di Tecnologia
* Author: Ugo Pattacini
* email:  ugo.pattacini@iit.it
* Permission is granted to copy, distribute, and/or modify this program
* under the terms of the GNU General Public License, version 2 or any
* later version published by the Free Software Foundation.
*
* A copy of the license can be found at
* http://www.robotcub.org/icub/license/gpl.txt
*
* This program is distributed in the hope that it will be useful, but
* WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
* Public License for more details
*/

#include <cmath>
#include <deque>
#include <string>
#include <iostream>
#include <iomanip>
#include <fstream>

#include <yarp/os/all.h>
#include <yarp/sig/all.h>
#include <yarp/math/Math.h>
#include <yarp/math/RandnScalar.h>

#include <IpTNLP.hpp>
#include <IpIpoptApplication.hpp>

using namespace std;
using namespace yarp::os;
using namespace yarp::sig;
using namespace yarp::math;


/****************************************************************/
class SphereNLP : public Ipopt::TNLP
{
protected:
    const int opt;
    const deque<Vector> &points;
    Vector result;

public:
    /****************************************************************/
    SphereNLP(const int opt_, const deque<Vector> &points_) :
              opt(opt_), points(points_), result(6,0.0) { }

    /****************************************************************/
    Vector get_result() const
    {
        return result;
    }

    /****************************************************************/
    bool get_nlp_info(Ipopt::Index &n,Ipopt::Index &m,Ipopt::Index &nnz_jac_g,
                      Ipopt::Index &nnz_h_lag,IndexStyleEnum &index_style)
    {
        n=6;
        if (opt==1)
        {
            m=1;
            nnz_jac_g=6;
        }
        else
        {
            m=0;
            nnz_jac_g=0;
        }
        nnz_h_lag=0;
        index_style=TNLP::C_STYLE;
        return true;
    }

    /****************************************************************/
    bool get_bounds_info(Ipopt::Index n,Ipopt::Number *x_l,Ipopt::Number *x_u,
                         Ipopt::Index m,Ipopt::Number *g_l,Ipopt::Number *g_u)
    {
        x_l[0]=-10.0; x_u[0]=10.0;
        x_l[1]=-10.0; x_u[1]=10.0;
        x_l[2]=-10.0; x_u[2]=10.0;
        x_l[3]=0.001; x_u[3]=1e9;
        x_l[4]=0.001; x_u[4]=1e9;
        x_l[5]=0.001; x_u[5]=1e9;
        if (opt==1)
        {
            g_l[0]=0.0;
            g_u[0]=1e-3;
        }
        return true;
    }

    /****************************************************************/
    bool get_starting_point(Ipopt::Index n,bool init_x,Ipopt::Number *x,
                            bool init_z,Ipopt::Number *z_L,Ipopt::Number *z_U,
                            Ipopt::Index m,bool init_lambda,Ipopt::Number *lambda)
    {
        x[0]=0.0;
        x[1]=0.0;
        x[2]=0.0;
        x[3]=10.0;
        x[4]=10.0;
        x[5]=10.0;
        return true;
    }

    /****************************************************************/
    bool eval_f(Ipopt::Index n,const Ipopt::Number *x,bool new_x,
                Ipopt::Number &obj_value)
    {
        if (opt==1)
            obj_value=x[3]*x[4]*x[5];
        else
        {
            Vector c(3);
            c[0]=x[0];
            c[1]=x[1];
            c[2]=x[2];

            Matrix A=zeros(3,3);
            A(0,0)=1.0/(x[3]*x[3]);
            A(1,1)=1.0/(x[4]*x[4]);
            A(2,2)=1.0/(x[5]*x[5]);

            obj_value=0.0;
            double coeff=x[3]*x[4]*x[5];
            for (size_t i=0; i<points.size(); i++)
            {
                Vector p=points[i]; p-=c;
                double tmp=dot(p,A*p)-1.0;
                obj_value+=coeff*tmp*tmp;
            }
            obj_value/=points.size();
        }
        return true;
    }

    /****************************************************************/
    bool eval_grad_f(Ipopt::Index n,const Ipopt::Number* x,bool new_x,
                     Ipopt::Number *grad_f)
    {
        if (opt==1)
        {
            grad_f[0]=0.0;
            grad_f[1]=0.0;
            grad_f[2]=0.0;
            grad_f[3]=x[4]*x[5];
            grad_f[4]=x[3]*x[5];
            grad_f[5]=x[3]*x[4];
        }
        else
        {
            Vector c(3);
            c[0]=x[0];
            c[1]=x[1];
            c[2]=x[2];

            Matrix A=zeros(3,3);
            A(0,0)=1.0/(x[3]*x[3]);
            A(1,1)=1.0/(x[4]*x[4]);
            A(2,2)=1.0/(x[5]*x[5]);

            Vector d0(3,0.0),d1(3,0.0),d2(3,0.0);
            d0[0]=d1[1]=d2[2]=1.0;

            Matrix D3=zeros(3,3);
            D3(0,0)=-2.0/(x[3]*x[3]*x[3]);

            Matrix D4=zeros(3,3);
            D4(1,1)=-2.0/(x[4]*x[4]*x[4]);

            Matrix D5=zeros(3,3);
            D5(2,2)=-2.0/(x[5]*x[5]*x[5]);

            for (Ipopt::Index i=0; i<n; i++)
                grad_f[i]=0.0;

            double coeff=x[3]*x[4]*x[5];
            for (size_t i=0; i<points.size(); i++)
            {
                Vector p=points[i]; p-=c;
                double tmp=dot(p,A*p)-1.0;                
                double tmp1=tmp*tmp;
                double tmp2=2.0*tmp;
                grad_f[0]-=coeff*tmp2*(dot(d0,A*p)+dot(p,A*d0));
                grad_f[1]-=coeff*tmp2*(dot(d1,A*p)+dot(p,A*d1));
                grad_f[2]-=coeff*tmp2*(dot(d2,A*p)+dot(p,A*d2));
                grad_f[3]+=coeff*tmp2*dot(p,D3*p)+x[4]*x[5]*tmp1;
                grad_f[4]+=coeff*tmp2*dot(p,D4*p)+x[3]*x[5]*tmp1;
                grad_f[5]+=coeff*tmp2*dot(p,D5*p)+x[3]*x[4]*tmp1;
            }

            for (Ipopt::Index i=0; i<n; i++)
                grad_f[i]/=points.size();
        }
        return true;
    }

    /****************************************************************/
    bool eval_g(Ipopt::Index n,const Ipopt::Number *x,bool new_x,
                Ipopt::Index m,Ipopt::Number *g)
    {
        if (opt==1)
        {
            Vector c(3);
            c[0]=x[0];
            c[1]=x[1];
            c[2]=x[2];

            Matrix A=zeros(3,3);
            A(0,0)=1.0/(x[3]*x[3]);
            A(1,1)=1.0/(x[4]*x[4]);
            A(2,2)=1.0/(x[5]*x[5]);

            g[0]=0.0;
            for (size_t i=0; i<points.size(); i++)
            {
                Vector p=points[i]; p-=c;
                double tmp=dot(p,A*p)-1.0;
                g[0]+=tmp*tmp;
            }
            g[0]/=points.size();
        }
        return true;
    }

    /****************************************************************/
    bool eval_jac_g(Ipopt::Index n,const Ipopt::Number *x,bool new_x,
                    Ipopt::Index m,Ipopt::Index nele_jac,Ipopt::Index *iRow,
                    Ipopt::Index *jCol,Ipopt::Number *values)
    {
        if (opt==1)
        {
            if (values==NULL)
            {
                iRow[0]=0; jCol[0]=0;
                iRow[1]=0; jCol[1]=1;
                iRow[2]=0; jCol[2]=2;
                iRow[3]=0; jCol[3]=3;
                iRow[4]=0; jCol[4]=4;
                iRow[5]=0; jCol[5]=5;
            }
            else
            {
                Vector c(3);
                c[0]=x[0];
                c[1]=x[1];
                c[2]=x[2];

                Matrix A=zeros(3,3);
                A(0,0)=1.0/(x[3]*x[3]);
                A(1,1)=1.0/(x[4]*x[4]);
                A(2,2)=1.0/(x[5]*x[5]);

                Vector d0(3,0.0),d1(3,0.0),d2(3,0.0);
                d0[0]=d1[1]=d2[2]=1.0;

                Matrix D3=zeros(3,3);
                D3(0,0)=-2.0/(x[3]*x[3]*x[3]);

                Matrix D4=zeros(3,3);
                D4(1,1)=-2.0/(x[4]*x[4]*x[4]);

                Matrix D5=zeros(3,3);
                D5(2,2)=-2.0/(x[5]*x[5]*x[5]);

                for (Ipopt::Index i=0; i<n; i++)
                    values[i]=0.0;

                for (size_t i=0; i<points.size(); i++)
                {
                    Vector p=points[i]; p-=c;
                    double tmp=2.0*(dot(p,A*p)-1.0);
                    values[0]-=tmp*(dot(d0,A*p)+dot(p,A*d0));
                    values[1]-=tmp*(dot(d1,A*p)+dot(p,A*d1));
                    values[2]-=tmp*(dot(d2,A*p)+dot(p,A*d2));
                    values[3]+=tmp*dot(p,D3*p);
                    values[4]+=tmp*dot(p,D4*p);
                    values[5]+=tmp*dot(p,D5*p);
                }

                for (Ipopt::Index i=0; i<n; i++)
                    values[i]/=points.size();
            }
        }
        return true;
    }

    /****************************************************************/
    bool eval_h(Ipopt::Index n,const Ipopt::Number *x,bool new_x,
                Ipopt::Number obj_factor,Ipopt::Index m,const Ipopt::Number *lambda,
                bool new_lambda,Ipopt::Index nele_hess,Ipopt::Index *iRow,
                Ipopt::Index *jCol,Ipopt::Number *values)
    {
        return true;
    }

    /****************************************************************/
    void finalize_solution(Ipopt::SolverReturn status,Ipopt::Index n,
                           const Ipopt::Number *x,const Ipopt::Number *z_L,
                           const Ipopt::Number *z_U,Ipopt::Index m,
                           const Ipopt::Number *g,const Ipopt::Number *lambda,
                           Ipopt::Number obj_value,const Ipopt::IpoptData *ip_data,
                           Ipopt::IpoptCalculatedQuantities *ip_cq)
    {
        for (Ipopt::Index i=0; i<n; i++)
            result[i]=x[i];
    }
};


/****************************************************************/
int main(int argc, char *argv[])
{
    ResourceFinder rf;
    rf.configure(argc,argv);
    int opt=rf.check("opt",Value(0)).asInt();
    double noise=rf.check("noise",Value(0.0)).asDouble();

    Vector c0(3);
    c0[0]=1.0;
    c0[1]=2.0;
    c0[2]=3.0;
    double a=0.05;
    double b=0.06;
    double c=0.07;

    RandnScalar n;
    n.init();

    deque<Vector> points;
    for (double theta=0.0; theta<M_PI; theta+=10.0*(M_PI/180.0))
    {
        for (double phi=0.0; phi<M_PI/2.0; phi+=10.0*(M_PI/180.0))
        {
            Vector p=c0;
            p[0]+=a*cos(theta)*sin(phi)+n.get(0.0,noise);
            p[1]+=b*sin(theta)*sin(phi)+n.get(0.0,noise);
            p[2]+=c*cos(phi)+n.get(0.0,noise);
            points.push_back(p);
        }
    }

    Ipopt::SmartPtr<Ipopt::IpoptApplication> app=new Ipopt::IpoptApplication;
    if (opt==1)
    {
        app->Options()->SetNumericValue("tol",1e-5);
        app->Options()->SetNumericValue("constr_viol_tol",1e-4);
        app->Options()->SetIntegerValue("acceptable_iter",5);
        app->Options()->SetNumericValue("acceptable_tol",3e-4);
    }
    else
    {
        app->Options()->SetNumericValue("tol",1e-8);
        app->Options()->SetIntegerValue("acceptable_iter",0);
    }
    app->Options()->SetStringValue("mu_strategy","monotone");
    app->Options()->SetIntegerValue("max_iter",2000);
    app->Options()->SetStringValue("hessian_approximation","limited-memory");
    app->Options()->SetStringValue("derivative_test","first-order");
    app->Options()->SetIntegerValue("print_level",5);
    app->Initialize();

    Ipopt::SmartPtr<SphereNLP> nlp=new SphereNLP(opt,points);
    Ipopt::ApplicationReturnStatus status=app->OptimizeTNLP(GetRawPtr(nlp));
    Vector result=nlp->get_result();

    Vector center=result.subVector(0,2);
    Vector radii=result.subVector(3,5);

    Matrix A=zeros(3,3);
    A(0,0)=1.0/(radii[0]*radii[0]);
    A(1,1)=1.0/(radii[1]*radii[1]);
    A(2,2)=1.0/(radii[2]*radii[2]);

    double error=0.0;
    for (size_t i=0; i<points.size(); i++)
    {
        Vector p=points[i]; p-=center;
        double tmp=dot(p,A*p)-1.0;
        error+=tmp*tmp;
    }
    error=sqrt(error)/points.size();

    cout<<"center=("<<center.toString(5,5)<<")"<<endl;
    cout<<"radii=("<<radii.toString(5,5)<<")"<<endl;
    cout<<"error="<<error<<endl;
    
    ofstream fout;
    fout.open("points.off");
    fout<<"COFF"<<endl;
    fout<<points.size()<<" 0 0"<<endl;
    for (size_t i=0; i<points.size(); i++)
        fout<<points[i][0]<<" "
            <<points[i][1]<<" "
            <<points[i][2]<<" "
            <<"255 0 0"
            <<endl;
    fout.close();

    points.clear();
    for (double theta=0.0; theta<2.0*M_PI; theta+=10.0*(M_PI/180.0))
    {
        for (double phi=0.0; phi<M_PI; phi+=10.0*(M_PI/180.0))
        {
            Vector p=center;
            p[0]+=radii[0]*cos(theta)*sin(phi);
            p[1]+=radii[1]*sin(theta)*sin(phi);
            p[2]+=radii[2]*cos(phi);
            points.push_back(p);
        }
    }

    fout.open("ellipsoid.off");
    fout<<"COFF"<<endl;
    fout<<points.size()<<" 0 0"<<endl;
    for (size_t i=0; i<points.size(); i++)
        fout<<points[i][0]<<" "
        <<points[i][1]<<" "
        <<points[i][2]<<" "
        <<"100 100 150"
        <<endl;
    fout.close();
    
    return 0;
}
