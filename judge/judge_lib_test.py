# Copyright 2021 The Cirq Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List

import cirq
import numpy as np

from judge.judge_lib import score_input, JudgeLogEntry


def test_passing_case():
    def m(qs: List[cirq.GridQubit], matrix: np.ndarray) -> cirq.OP_TREE:
        return [], []

    entry = JudgeLogEntry(task="hello")
    score_input(m, np.eye(2), entry, 2, 1, 0)

    expected = """executing method (0 pts): ✔ [0 pts]
    Close in trace distance (2 pts): ✔ [2 pts]
    Circuit structure (4 pts): ✔ [4 pts]
    - 2-qubit gates in your result: 0
    - Lower bound for general case: 0
    Valid for Sycamore device (2 pts): ✔ [2 pts]"""

    _assert_lines(entry, expected)


def test_three_qubit_gate():
    def m(qs: List[cirq.GridQubit], matrix: np.ndarray) -> cirq.OP_TREE:
        return [cirq.CCX(*qs)], []

    entry = JudgeLogEntry(task="hello")
    score_input(
        m,
        input=cirq.unitary(cirq.CCX),
        result=entry,
        multiplier=2,
        n_qubits=3,
        min_two_qubit=0,
    )

    expected = """executing method (0 pts): ✔ [0 pts]
    2+ qubit gates (0 pts): ✘
    Number of gates that need more than two qubits: 1 <-- it should be zero!
    Close in trace distance (2 pts): ✘
    Circuit structure (4 pts): ✘     
    Valid for Sycamore device (2 pts): ✘"""

    _assert_lines(entry, expected)


def test_sneaky_three_qubit_gate():
    class SneakyThreeQubit(cirq.Gate):
        def _num_qubits_(self) -> int:
            return 2

        def _unitary_(self):
            return cirq.unitary(cirq.CCX)

    def m(qs: List[cirq.GridQubit], matrix: np.ndarray) -> cirq.OP_TREE:
        return [SneakyThreeQubit().on(qs[0], qs[1])], []

    entry = JudgeLogEntry(task="hello")

    score_input(
        m,
        input=cirq.unitary(cirq.CCX),
        result=entry,
        multiplier=2,
        n_qubits=3,
        min_two_qubit=0,
    )

    expected = """executing method (0 pts): ✘
    <class 'ValueError'>: cannot reshape array of size 64 into shape (2,2,2,
    2+ qubit gates (0 pts): ✘
    Close in trace distance (2 pts): ✘
    Circuit structure (4 pts): ✘        
    Valid for Sycamore device (2 pts): ✘"""

    _assert_lines(entry, expected)


def test_fail_on_second():
    def m(qs: List[cirq.GridQubit], matrix: np.ndarray) -> cirq.OP_TREE:
        return [cirq.GlobalPhaseOperation(-1)], []

    entry = JudgeLogEntry(task="hello")
    score_input(
        m, input=np.eye(2), result=entry, multiplier=2, n_qubits=1, min_two_qubit=0
    )

    expected = """executing method (0 pts): ✔ [0 pts]
    2+ qubit gates (0 pts): ✔ [0 pts]
    Close in trace distance (2 pts): ✔ [2 pts]
     Circuit structure (4 pts): ✔ [4 pts]
     - 2-qubit gates in your result: 0
     - Lower bound for general case: 0       
    Valid for Sycamore device (2 pts): ✘
    <class 'ValueError'>: -1 is not a supported gate"""

    _assert_lines(entry, expected)


def test_fail_on_global_phase():
    def m(qs: List[cirq.GridQubit], matrix: np.ndarray) -> cirq.OP_TREE:
        return [cirq.X(*qs)], []

    entry = JudgeLogEntry(task="hello")
    score_input(
        m, input=np.eye(2), result=entry, multiplier=2, n_qubits=1, min_two_qubit=0
    )

    expected = """executing method (0 pts): ✔
    2+ qubit gates (0 pts): ✔
    Close in trace distance (2 pts): ✘
    <class 'AssertionError'>:
     Circuit structure (4 pts): ✘         
    Valid for Sycamore device (2 pts): ✘
"""

    _assert_lines(entry, expected)


def test_error_in_method():
    def m(qs: List[cirq.GridQubit], matrix: np.ndarray) -> cirq.OP_TREE:
        raise ValueError("bla")

    entry = JudgeLogEntry(task="hello")
    score_input(
        m, input=np.eye(2), result=entry, multiplier=2, n_qubits=1, min_two_qubit=0
    )

    expected = """
    executing method (0 pts): ✘
    ValueError
    2+ qubit gates (0 pts): ✘
    Close in trace distance (2 pts): ✘        
     Circuit structure (4 pts): ✘
    Valid for Sycamore device (2 pts): ✘"""

    _assert_lines(entry, expected)


def test_notimplemented_in_method():
    def m(qs: List[cirq.GridQubit], matrix: np.ndarray) -> cirq.OP_TREE:
        return NotImplemented, []

    entry = JudgeLogEntry(task="hello")
    score_input(
        m, input=np.eye(2), result=entry, multiplier=2, n_qubits=1, min_two_qubit=0
    )

    expected = """
    executing method (0 pts): ✔
    2+ qubit gates (0 pts): [skipped]
    Close in trace distance (2 pts): [skipped]
    Circuit structure (4 pts): [skipped]
    Valid for Sycamore device (2 pts): [skipped]"""

    _assert_lines(entry, expected)


def _assert_lines(entry, expected):
    actual_lines = entry.msgs.splitlines()
    i = -1
    for l in expected.splitlines():
        found = False
        while i < len(actual_lines) and not found:
            i += 1
            found = i < len(actual_lines) and l.strip(" ") in actual_lines[i]

        assert found, f"can't find '{l}' in this order! Messages:\n{entry.msgs}"
