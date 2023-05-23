# pyavanimate/setup.py
#
# Copyright 2023 Winton Davies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from setuptools import setup
desc = \
'''
simple API to turn a ChatGPT prompt into a list
of valid songs and metadata.
'''

setup(
        name='chatspot',
        version='0.0.1',
        description=desc,
        url='git@github.com:davies-w/chatspot.git',
        author='Winton Davies',
        author_email='wdavies@cs.stanford.edu',
        install_requires=["openai", "spotipy"],
        packages=['chatspot'],
        zip_safe=True
    )
