#!/usr/bin/env bash

devpi-server --start --init --host=0.0.0.0 --debug
tail -f /devpi/.xproc/devpi-server/xprocess.log
