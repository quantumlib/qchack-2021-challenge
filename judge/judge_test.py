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

import pytest
from _pytest.outcomes import Skipped

import cirq
import numpy as np
import traceback
from judge.judge_lib import JudgeLog, JudgeLogEntry, score_input


@pytest.fixture(scope="session")
def judge_log():
    log = JudgeLog()
    yield log
    print(log.results())


def test_simple_identity(judge_log):
    result = JudgeLogEntry(task="Simple identity check.")
    _score_and_log(np.eye(2), judge_log, 1, result, n_qubits=1, min_two_qubit=0)


@pytest.mark.parametrize(
    "gate",
    [
        cirq.X,
        cirq.Y,
        cirq.Z,
        cirq.H,
        cirq.S,
        cirq.T,
    ],
)
def test_single_qubit_gates(judge_log, gate):
    multiplier = 2
    result = JudgeLogEntry(task=f"Single qubit gate {gate}")
    qs = cirq.LineQubit.range(1)
    input = cirq.unitary(gate(*qs))

    _score_and_log(input, judge_log, multiplier, result, n_qubits=1, min_two_qubit=0)


@pytest.mark.parametrize(
    "gate",
    [
        cirq.google.SycamoreGate(),
        cirq.CX,
        cirq.XX,
        cirq.YY,
        cirq.ZZ,
        cirq.IdentityGate(num_qubits=2),
    ],
)
def test_two_qubit_gates(judge_log, gate):
    multiplier = 4
    result = JudgeLogEntry(task=f"Two-qubit gate {gate}")

    input = cirq.unitary(gate(*(cirq.LineQubit.range(2))))

    _score_and_log(input, judge_log, multiplier, result, n_qubits=2, min_two_qubit=1)


@pytest.mark.parametrize(
    "gate",
    [
        cirq.CCX,
        cirq.CSWAP,
        cirq.ControlledGate(cirq.ISWAP ** 0.5),
        cirq.CCZ,
        cirq.IdentityGate(num_qubits=3),
    ],
)
def test_three_qubit_gates(judge_log, gate):
    multiplier = 8
    result = JudgeLogEntry(task=f"Three-qubit gate {gate}")

    input = cirq.unitary(gate(*(cirq.LineQubit.range(3))))

    _score_and_log(input, judge_log, multiplier, result, n_qubits=3)


@pytest.mark.parametrize(
    "gate",
    [
        cirq.ControlledGate(cirq.CCX),
        cirq.IdentityGate(num_qubits=4),
    ],
)
def test_four_qubit_gates(judge_log, gate):
    multiplier = 16
    result = JudgeLogEntry(task=f"Four-qubit gate {gate}")

    input = cirq.unitary(gate(*(cirq.LineQubit.range(4))))

    _score_and_log(input, judge_log, multiplier, result, n_qubits=4)


@pytest.mark.parametrize("n_qubits", [n for n in range(5, 10)])
def test_identities(judge_log, n_qubits):
    multiplier = 1
    gate = cirq.IdentityGate(num_qubits=n_qubits)
    result = JudgeLogEntry(task=f"{n_qubits}-qubit identity gate {gate}")

    input = cirq.unitary(gate(*(cirq.LineQubit.range(n_qubits))))

    _score_and_log(input, judge_log, multiplier, result, n_qubits, min_two_qubit=0)


@pytest.mark.parametrize("n_qubits", [n for n in range(1, 9)])
def test_random_unitaries(judge_log, n_qubits):
    # TODO do some random testing with different circuit densities
    multiplier = 2 ** n_qubits * 2
    s = np.random.RandomState(seed=12312422)

    result = JudgeLogEntry(task=f"{n_qubits}-qubit random unitary")

    input = cirq.testing.random_unitary(2 ** n_qubits, random_state=s)

    _score_and_log(input, judge_log, multiplier, result, n_qubits)


@pytest.mark.parametrize("n_qubits", [n for n in range(1, 9)])
def test_random_diagonals(judge_log, n_qubits):
    s = np.random.RandomState(seed=42)
    multiplier = 2 ** n_qubits
    result = JudgeLogEntry(task=f"{n_qubits}-qubit random diagonal unitary")

    angles = [angle * 2 * np.pi for angle in s.rand(2 ** n_qubits)]
    input = np.diag([np.exp(1j * angle) for angle in angles])

    _score_and_log(input, judge_log, multiplier, result, n_qubits)


@pytest.mark.parametrize("n_qubits", [n for n in range(1, 9)])
def test_incrementers(judge_log, n_qubits):
    multiplier = 2 ** n_qubits
    result = JudgeLogEntry(task=f"{n_qubits}-qubit incrementer")

    input = np.empty((2 ** n_qubits, 2 ** n_qubits))
    input[1:] = np.eye(2 ** n_qubits)[:-1]
    input[:1] = np.eye(2 ** n_qubits)[-1:]

    _score_and_log(input, judge_log, multiplier, result, n_qubits)


def _score_and_log(
    input, judge_log, multiplier, result, n_qubits, min_two_qubit=np.inf
):
    print(result.title())
    try:
        from solution import matrix_to_sycamore_operations

        score_input(
            matrix_to_sycamore_operations,
            input,
            result,
            multiplier,
            n_qubits,
            min_two_qubit,
        )
    except Skipped:
        result.msgs += "skipped\n"
    except BaseException as ex:
        result.msgs += f"✘\n {type(ex)}: {str(ex)}" f"\n {traceback.format_exc()}"
    finally:
        judge_log.entries.append(result)
        print(result)
