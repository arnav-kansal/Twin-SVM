#!/usr/bin/env python

import numpy as np

from numpy import linalg
from cvxopt import solvers,matrix

#__copyright__ = ""
#__license__ = "GPL"
# __version__ = "1.1"
# __maintainer__ = "Arnav Kansal"
# __email__ = "ee1130440@ee.iitd.ac.in"
# __status__ = "Production"


def Twin_plane_1(R,S,C1,Epsi1,regulz1):
	StS = np.dot(S.T,S)
	# for regularization we add identity matrix with wt. before inversion
	StS = StS + regulz1*(np.identity(StS.shape[0]))
	StSRt = linalg.solve(StS,R.T)
	RtStSRt = np.dot(R,StSRt)
	RtStSRt = (RtStSRt+(RtStSRt.T))/2
	m2 = R.shape[0]
	e2 = -np.ones((m2,1))
	solvers.options['show_progress'] = False
	vlb = np.zeros((m2,1))
	vub = C1*(np.ones((m2,1)))
	# x<=vub
	# x>=vlb -> -x<=-vlb
	# cdx<=vcd
	cd = np.vstack((np.identity(m2),-np.identity(m2)))
	vcd = np.vstack((vub,-vlb))
	alpha = solvers.qp(matrix(RtStSRt,tc='d'),matrix(e2,tc='d'),matrix(cd,tc='d'),matrix(vcd,tc='d'))#,matrix(0.0,(1,m1)),matrix(0.0))#,None,matrix(x0))
	alphasol = np.array(alpha['x'])
	z = -np.dot(StSRt,alphasol)
	w1 = z[:len(z)-1]
	b1 = z[len(z)-1]
	return [w1,b1]