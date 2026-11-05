# -*- coding: utf-8 -*-
#
# Copyright (C) 2026 Zoltan Doczi
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
"""Conformal PEC: operator-level checks through the XML attribute and the openEMS binary (no time stepping)."""

import os
import shutil
import subprocess
import tempfile
import unittest

import numpy as np

from CSXCAD import ContinuousStructure
from openEMS.openEMS import openEMS


def _model(angle, conformal):
    """ a 10 mm x 2 mm metal strip, rotated by angle (deg) in the z = 0 plane, on a 0.5 mm mesh """
    fdtd = openEMS(NrTS=10, ConformalPEC=1 if conformal else 0)
    fdtd.SetGaussExcite(1e9, 1e9)
    fdtd.SetBoundaryCond(['PEC'] * 6)
    csx = ContinuousStructure()
    fdtd.SetCSX(csx)
    g = csx.GetGrid()
    g.SetDeltaUnit(1e-3)
    for ax in 'xy':
        g.SetLines(ax, np.arange(-10, 10.01, 0.5))
    g.SetLines('z', [-3, -1.5, 0, 1.5, 3])
    c, s = np.cos(np.radians(angle)), np.sin(np.radians(angle))
    pts = np.array([(c * x - s * y, s * x + c * y) for x, y in ((-5, -1), (5, -1), (5, 1), (-5, 1))])
    csx.AddMetal('strip').AddLinPoly(points=pts.T, norm_dir=2, elevation=0, length=0)
    return fdtd


def _pec_counts(fdtd, path):
    """ (PEC edges, conformal edges) reported by the solver's setup for the model """
    xml = os.path.join(path, 'model.xml')
    fdtd.Write2XML(xml)
    exe = shutil.which('openEMS') or os.environ.get('OPENEMS_BIN', '')
    if not exe:
        raise unittest.SkipTest('openEMS binary not on PATH')
    out = subprocess.run([exe, xml, '--no-simulation', '-v'], cwd=path, capture_output=True, text=True).stdout
    pec = conf = 0
    for line in out.splitlines():
        if line.startswith('Number of PEC edges'):
            pec = int(line.split(':')[1])
        if line.startswith('Conformal PEC edges'):
            conf = int(line.split(':')[1].split()[0])
    return pec, conf


class Test_ConformalPEC(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def test_kwarg_and_attribute(self):
        xml = os.path.join(self.tmp.name, 'a.xml')
        _model(0, True).Write2XML(xml)
        self.assertIn('ConformalPEC="1"', open(xml).read())
        _model(0, False).Write2XML(xml)
        self.assertNotIn('ConformalPEC', open(xml).read())

    def test_aligned_strip_has_no_partial_edges(self):
        # every edge of a mesh-aligned strip is either fully metal or free: the conformal model changes nothing
        pec0, conf0 = _pec_counts(_model(0, False), self.tmp.name)
        pec1, conf1 = _pec_counts(_model(0, True), self.tmp.name)
        self.assertEqual(conf0, 0)
        self.assertEqual(conf1, 0)
        self.assertEqual(pec0, pec1)
        self.assertGreater(pec0, 0)

    def test_rotated_strip_has_partial_edges(self):
        # a strip at 30 deg cuts through cells: edges partly inside metal are counted, fully covered ones stay PEC
        pec0, conf0 = _pec_counts(_model(30, False), self.tmp.name)
        pec1, conf1 = _pec_counts(_model(30, True), self.tmp.name)
        self.assertEqual(conf0, 0)
        self.assertGreater(conf1, 0)
        self.assertGreater(pec1, 0)
        self.assertLessEqual(pec1, pec0)       # the midpoint rule counted some partly covered edges as metal


if __name__ == '__main__':
    unittest.main()
