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


def Twin_plane_2(L,N,C2,Epsi2,regulz2):
	NtN = np.dot(N.T,N)
	# for regularization we add identity matrix with wt. before inversion
	NtN = NtN + regulz2*(np.identity(NtN.shape[0]))
	NtNLt = linalg.solve(NtN,L.T)
	LtNtNLt = np.dot(L,NtNLt)
	LtNtNLt = (LtNtNLt+(LtNtNLt.T))/2
	m1 = L.shape[0]
	e1 = -np.ones((m1,1))
	solvers.options['show_progress'] = False
	vlb = np.zeros((m1,1))
	vub = C2*(np.ones((m1,1)))
	# x<=vub
	# x>=vlb -> -x<=-vlb
	# cdx<=vcd
	cd = np.vstack((np.identity(m1),-np.identity(m1)))
	vcd = np.vstack((vub,-vlb))
	gamma = solvers.qp(matrix(LtNtNLt,tc='d'),matrix(e1,tc='d'),matrix(cd,tc='d'),matrix(vcd,tc='d'))#,matrix(0.0,(1,m1)),matrix(0.0))#,None,matrix(x0))
	gammasol = np.array(gamma['x'])
	z = -np.dot(NtNLt,gammasol)
	w2 = z[:len(z)-1]
	b2 = z[len(z)-1]
	return [w2,b2]