# Qchack 2021 Google Challenge 

This is a challenge for the brave 2021 [qchack.io](https://qchack.io) participants.

## Instructions

Hello, intrepid qchacker, welcome to the <G|oogl|e> challenge! 

**Background**

In quantum computing, the gate model plays a central role. Today most quantum algorithm designers express quantum algorithms via things called quantum gates. Quantum hardware, like the Google Sycamore device can execute only certain types of gates, and only one and two-qubit gates. However, this means that multi-qubit operations need to be compiled down to the supported gates. Also, another complication is that devices have restrictions - not all qubits are connected to each other!


**Challenge**

Your challenge is to compile an n-qubit (1<=n<=8) unitary matrix (<img src="https://render.githubusercontent.com/render/math?math=2^n \times 2^n"> unitary matrix) to a list of operations that can run on the Google Sycamore device - i.e. to implement this method: 

```python
from typing import List, Tuple

import numpy as np
import cirq

def matrix_to_sycamore_operations(target_qubits: List[cirq.GridQubit], matrix: np.ndarray) -> Tuple[cirq.OP_TREE, List[cirq.GridQubit]]:
    """ A method to convert a unitary matrix to a list of Sycamore operations. 
    
    This method will return a list of `cirq.Operation`s using the qubits and (optionally) ancilla 
    qubits to implement the unitary matrix `matrix` on the target qubits `qubits`. 
    The operations are also supported by `cirq.google.gate_sets.SYC_GATESET`. 

    Args:
        target_qubits: list of qubits the returned operations will act on. The qubit order defined by the list 
            is assumed to be used by the operations to implement `matrix`.
        matrix: a matrix that is guaranteed to be unitary and of size (2**len(qs), 2**len(qs)).
    Returns: 
        A tuple of operations and ancilla qubits allocated. 
            Operations: In case the matrix is supported, a list of operations `ops` is returned. 
                `ops` acts on `qs` qubits and for which `cirq.unitary(ops)` is equal to `matrix` up 
                 to certain tolerance. In case the matrix is not supported, it might return NotImplemented to 
                 reduce the noise in the judge output.
            Ancilla qubits: In case ancilla qubits are allocated a list of ancilla qubits. Otherwise 
                an empty list.
        .   
    """
    return NotImplemented, []
```

**Input unitaries**

Your method will be tested against different inputs, which will be 1-8 qubit input unitaries. 
Some of the unitaries will have structure that you should leverage to create efficient circuits, this can give you more points!
Some of them will be completely random.  

**Output**

Your method will return a two-tuple of two things:

- list of operations - these operations using the Sycamore gateset (see more details in additional information) will be used to create a `cirq.Circuit` and we'll take the unitary matrix of it to compare it with the input as explained below. 
- ancilla qubits (advanced) - list of ancilla qubits, e.g. `[]` means you didn't allocate any ancillae, `[cirq.GridQubit(2,3)]` means you allocated a single ancilla qubit 

**Ancilla qubits** 

Some unitaries will benefit from using ancilla qubits. 
You can allocate ancilla qubits, as long as the total number of qubits is less than or equal to 10 qubits.

The expected unitary is the tensor product of the input matrix and the identity on the ancilla qubits:

```
expected_unitary = cirq.kron(input, np.eye(2 ** len(ancillae)))
```
  
Where `input` is the original input. 

**Qubit order**

Qubit ordering is determined by the passed in `target_qubits` list and the order of the returned ancilla qubits. 
This is passed in to the `circuit.unitary()` method, i.e. given your response solution we will evaluate the unitary matrix of your operation the following way:
 
```
response, ancillae = solution.matrix_to_sycamore_operations(target_qubits, input_unitary)
response_unitary = response_circuit.unitary(
                        qubit_order=qs + ancillae, 
                        qubits_that_should_be_present=qs + ancillae
                   )
```

Recall why this matters: for example, on two qubits, if the order is the regular big endian, the CNOT unitary is:

```
>>> a,b = cirq.LineQubit.range(2)
>>> cirq.Circuit(cirq.CNOT(a,b)).unitary(qubit_order=[a,b], qubits_that_should_be_present=[a,b])
array([[1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j]])
>>> cirq.Circuit(cirq.CNOT(a,b)).unitary(qubit_order=[b,a], qubits_that_should_be_present=[a,b])
array([[1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j]])
```


**Scoring**
 
We'll score each input separately, and sum them up. 

Scoring per input matrix: 

* you code *MUST* use the qubits that were passed in
* your code *MUST* execute otherwise no points are given 
* you *MUST* provide 1 and 2-qubit gates otherwise no points are given
* the diamond norm distance of your response *MUST* be close (max 1e-4) to the input unitary 
* the less two-qubit operations you return the more points you get
* you get extra points if the operations can run on Sycamore 
  

## Some additional information

Google's Sycamore device is a 54 qubit quantum chip:

```
>>> print(cirq.google.Sycamore)
                                             (0, 5)───(0, 6)
                                             │        │
                                             │        │
                                    (1, 4)───(1, 5)───(1, 6)───(1, 7)
                                    │        │        │        │
                                    │        │        │        │
                           (2, 3)───(2, 4)───(2, 5)───(2, 6)───(2, 7)───(2, 8)
                           │        │        │        │        │        │
                           │        │        │        │        │        │
                  (3, 2)───(3, 3)───(3, 4)───(3, 5)───(3, 6)───(3, 7)───(3, 8)───(3, 9)
                  │        │        │        │        │        │        │        │
                  │        │        │        │        │        │        │        │
         (4, 1)───(4, 2)───(4, 3)───(4, 4)───(4, 5)───(4, 6)───(4, 7)───(4, 8)───(4, 9)
         │        │        │        │        │        │        │        │
         │        │        │        │        │        │        │        │
(5, 0)───(5, 1)───(5, 2)───(5, 3)───(5, 4)───(5, 5)───(5, 6)───(5, 7)───(5, 8)
         │        │        │        │        │        │        │
         │        │        │        │        │        │        │
         (6, 1)───(6, 2)───(6, 3)───(6, 4)───(6, 5)───(6, 6)───(6, 7)
                  │        │        │        │        │
                  │        │        │        │        │
                  (7, 2)───(7, 3)───(7, 4)───(7, 5)───(7, 6)
                           │        │        │
                           │        │        │
                           (8, 3)───(8, 4)───(8, 5)
                                    │
                                    │
                                    (9, 4)

```

There are only a limited number of gates that this device supports. 

Suppose we pick these two adjacent qubits:

```
>>> a = cirq.GridQubit(3,5)
>>> b = cirq.GridQubit(3,6)
```

The supported gates on these qubits will be: 

- 1 qubit gates: 
  - `[X/Z/Y]PowGate`:  `cirq.google.Sycamore.validate_operation(cirq.X(a))`
  - `PhasedXZGate`:  `cirq.google.Sycamore.validate_operation(cirq.PhasedXZGate(x_exponent=1, z_exponent=1, axis_phase_exponent=1.2)(a))`
- 2 qubit gates: 
  - sqrt of ISWAP: `cirq.google.Sycamore.validate_operation(cirq.ISWAP(a,b)**0.5)`
  - SYC gate: `cirq.google.Sycamore.validate_operation(cirq.SYC(a,b))`

A Hint: before you jump in trying to figure out compilation to these gates, you might want to check our Cirq intro workshop or the Cirq reference for easier ways.

## How to participate? 

1. You will need an email address that you registered with for qchack. Only a single submission (the last one takes precedence) is accepted per email address! 
2. To get started, clone this repo: 

```bash
git clone github.com/quantumlib/qchack-2021-challenge
```
3. Write your code! You can work with a set of automated tests locally to check your progress. See [Testing](#Testing) for details!
4. When you are ready to submit, publish your code to a public Github/Gitlab/Bitbucket repo
5. Fill out [this Google form](https://forms.gle/N2M5PGmXwjZHr3SD9) with your repository URL. 

GOOD LUCK from the Google Quantum AI qchack.io team!


## Legal eligibility for prizes

Persons who are (1) residents of US embargoed countries, (2) ordinarily resident in US embargoed countries, or (3) otherwise prohibited by applicable export controls and sanctions programs may not receive prizes in this contest. We will also not be able to give out prizes for winners from the following countries: Russia, Ukraine, Kazakhstan, Belarus and Brazil. 

# How to ensure that my submission is valid?

## Do NOTs

The solution folder needs to stay the same.
Not following these instructions will result in a erroneous submission. 

- do not change `solution` folder name
- do not change the function name `matrix_to_sycamore_operations`
- do not change the function signature 

## Dos 

Ensure that the judge test in its current form runs on your code. 
That will ensure that the submission process will work! 

# Dependencies

The judge will have the dependencies installed during submission in judge/requirements.txt.
If your project requires _additional_ dependencies, only add the additional dependencies to solution/requirements.txt.  
If your project requires no additional dependencies, then just leave the solution/requirements.txt empty.

# Testing, scoring

You can run the judge yourself! This is provided so that you can track your progress. However, note that your final score might be different as we will use different randomization strategies and unitaries to test your code.
Scoring logic however will be similar to `score_input` in [judge/judge_lib.py](judge/judge_lib.py).
We recommend checking out the simpler test cases first to ensure getting some points! 


## Setup

Python3.8 is required, and the use of a virtual environment is recommended.

```bash
pip install -r judge/requirements.txt
```

## Running the judge 

Run the following to run all the tests once and get your full score: 

```bash
pytest judge/judge_test.py -rP
```

When you run this the first time, you should see something like this: 

```
...
------------------------------------------------------------------------------------------------- Captured stdout call -------------------------------------------------------------------------------------------------
/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\ [ 8-qubit incrementer ] /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\

executing method (0 pts): ✔ [0 pts]
2+ qubit gates (0 pts): [skipped] 
Close in trace distance (256 pts): [skipped] 
Circuit structure (512 pts): [skipped] 
Valid for Sycamore device (256 pts): [skipped] 
Result: 0.00 / 1024
----------------------------------------------------------------------------------------------- Captured stdout teardown -----------------------------------------------------------------------------------------------

====================================================================================================
Total score: 0.00 / 8616 points!

================================================================================================== 49 passed in 1.71s ==================================================================================================
```

Run the following if you want to run a particular test (note that the total score is only the score of the executed tests!):

```bash
pytest judge/judge_test.py -rP -k single_qubit
```

Run this to run all the tests in watch mode (anything you change triggers a rebuild):

```bash
ptw -- judge/judge_test.py -rP
```

# Support 

If you run into technical issues, feel free to file a ticket on this Github repo and or ping us on the Discord channel throughout the Hackathon.
 