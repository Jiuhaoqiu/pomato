.. ref-quick-start:

-------------------
Quick Start Guide
-------------------

This quick start guide will introduce the main concepts and functions of Pomato.


Creating Intializing a Mato Object
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The **MarketTool()** object provides the general API of Pomato's functionality.
It is typically initialized as::

    from pomato.market_tool import MarketTool

    mato = MarketTool()

The ``MarketTool`` object can be initialized with the following keyword arguments:

* ``opt_file = 'path-to-optfile'`` - references a option file in ``.json`` format. See below.
* ``model_horizon = rangeObject`` - limits the calculation horizon of the model to a subset of the available timesteps

Load data either from excel files::

    mato.load_data_from_file('test_data/diw_demo.xlsx')

or **MatPower** cases::

    mato.load_matpower_case('case118')

To set-up and run the model use::

    mato.init_market_model()
    mato.run_market_model()

(more options and features coming soon)


The Option File
^^^^^^^^^^^^^^^

The option file collects a set of keyword options in ``.json`` format to define the behavior of the model and the usage of the available data.

For example::

    {
        "opt": "ntc",
        "infeas_heat": true,
        "infeas_el": true
        "infeas_lines": true
    }

Keywords and their options:

* ``"opt"``:
    * ``"ntc"`` use net transfer capacity to solve the flows
    * ``"cbco"`` reduce line constraints to cbco matrix and solve security constrained dispatch
* ``"infeas_heat"``:
    * ``boolean``: Turn slack in heat energy-balances on or off
* ``"infeas_el"``:
    * ``boolean``: Turn slack in electricity energy-balances on or off
* ``"infeas_lines"``:
    * ``boolean``: Turn slack in flow energy-balances on or off


