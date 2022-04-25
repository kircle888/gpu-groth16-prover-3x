from json.tool import main
import os
import shutil
import subprocess
import time
import pandas as pd
from typing import Tuple, List, Set, Dict, Any
base_dir = os.getcwd()
data_dir = os.path.join(base_dir, "data")
if not os.path.exists(data_dir):
    os.mkdir(data_dir)
report_dir = os.path.join(base_dir, "reports")
if not os.path.exists(report_dir):
    os.mkdir(report_dir)
generator_path = os.path.join(base_dir, "generate_parameters")
assert os.path.exists(generator_path)
cudaprover_path = os.path.join(base_dir, "cuda_prover_piecewise")
assert os.path.exists(cudaprover_path)
main_path = os.path.join(base_dir, "main")
assert os.path.exists(main_path)


def call_foroutput(cmd: str, output_file=None, nonread=False) -> str:
    print("Executing:", cmd)
    if nonread and output_file is None:
        os.system(cmd)
    save_file = output_file if output_file is not None else "_stdout.txt"
    cmd = cmd+" | tee \"%s\"" % (save_file)
    os.system(cmd)
    if nonread:
        return None
    out = None
    with open(save_file, "r") as f:
        out = f.read()
    if output_file is None:
        os.remove(save_file)
    return out


class Profile:

    def __init__(self, data_name: str, method_name: str, profile: str) -> None:
        self.data_name: str = data_name
        self.method_name: str = method_name
        self.raw_profile: str = profile
        self.parsed, self.remain_lines = self._parse_colon(profile)
        self.parsed: Dict[str, str]
        self.remain_lines: List[str]
        self.base_data: pd.DataFrame = self._get_base_data(
            data_name, method_name, self.parsed, self.remain_lines)
        # print(self.base_data)
        self._detailed()

    @staticmethod
    def _parse_colon(profile: str) -> Tuple[dict, list]:
        lines = profile.splitlines()
        remain_lines = []
        parsed = dict()
        for line in lines:
            result = line.split(':')
            if len(result) == 2:
                parsed[result[0].strip()] = result[1].strip()
            elif line != "":
                remain_lines.append(line.strip())
        return parsed, remain_lines

    @staticmethod
    def _get_base_data(data_name, method_name, parsed_dict: Dict[str, str], remain_lines: List[str] = None) -> pd.DataFrame:
        data = dict()
        data['data_name'] = data_name
        data['method_name'] = method_name
        # data['unique_name'] = data_name+"-"+method_name
        data['GPU Launch'] = parsed_dict.get('gpu launch')
        data['CPU1'] = parsed_dict.get('cpu 1')
        data['GPU Time'] = parsed_dict.get('gpu e2e')
        data['Total Time'] = parsed_dict.get(
            'Total runtime (incl. file reads)')
        frm = pd.DataFrame([data])
        frm.set_index(['data_name', 'method_name'], inplace=True)
        return frm

    def _cpu_detailed(self) -> None:
        last_line = self.remain_lines[-1]
        assert last_line.startswith("(leave) Call to r1cs_gg_ppzksnark_prover")
        t = last_line.split('[')[1].split(']')[0]
        t = t.split('s x')[0]
        t = float(t)
        t = t*1000
        t = "%.0fms" % t
        self.base_data.loc[self.data_name].loc[self.method_name,
                                               'Total Time'] = t

    def _straus_detailed(self) -> None:
        pass

    def _pippenger_detailed(self) -> None:
        pass

    def _detailed(self) -> None:
        if self.method_name == "cpu":
            self._cpu_detailed()
        elif self.method_name == "straus":
            self._straus_detailed()
        elif self.method_name == "pippenger":
            self._pippenger_detailed()


class DataConfig:
    def __init__(self, curve: str, logN: int, test_times=1) -> None:
        self.curve = curve
        self.logN = logN
        self.test_times = test_times

    @property
    def N(self) -> int:
        return pow(2, self.logN)

    @property
    def name(self) -> str:
        return self.curve+"-"+str(self.logN)+"-"+str(self.test_times)

    def iname(self, i) -> str:
        return self.curve+"-"+str(self.logN)+"-"+str(i)

    @property
    def work_dir(self) -> str:
        return os.path.join(data_dir, self.name)

    @property
    def able(self) -> bool:
        if not os.path.exists(self.work_dir):
            return False
        os.chdir(self.work_dir)
        for i in range(0, self.test_times):
            if not os.path.exists("input-%d" % i):
                return False
            if not os.path.exists("params-%d" % i):
                return False
        os.chdir(base_dir)
        return True

    def generate(self) -> None:
        if not os.path.exists(self.work_dir):
            os.mkdir(self.work_dir)
        os.chdir(self.work_dir)
        with open("generate_log.txt", "w") as f:
            for i in range(0, self.test_times):
                t = time.time()
                if self.curve == "MNT6753":
                    os.system("%s %d %d" % (generator_path, 0, self.logN))
                elif self.curve == "MNT4753":
                    os.system("%s %d %d" % (generator_path, self.logN, 0))
                os.rename(self.curve+"-input", "input-%d" % i)
                os.rename(self.curve+"-parameters", "params-%d" % i)
                f.write("%f\n" % (time.time()-t))
        os.chdir(base_dir)

    @property
    def preprocessed_able(self) -> bool:
        if not self.able:
            return False
        os.chdir(self.work_dir)
        for i in range(0, self.test_times):
            if not os.path.exists("preprocessed-%d" % i):
                return False
        os.chdir(base_dir)
        return True

    def preprocess(self) -> None:
        assert self.able
        os.chdir(self.work_dir)
        with open("preprocess_log.txt", "w") as f:
            for i in range(0, self.test_times):
                t = time.time()
                os.system("%s %s preprocess params-%d" %
                          (main_path, self.curve, i))
                os.rename(self.curve+"_preprocessed", "preprocessed-%d" % i)
                f.write("%f\n" % (time.time()-t))
        os.chdir(base_dir)


class MethodConfig:
    def __init__(self, method="pippenger", C=7, R=32) -> None:
        self.method = method
        self.C = C
        self.R = R

    @property
    def checksum_filepath(self) -> str:
        if self.method == "cpu":
            return "checksum.txt"
        else:
            return "checksum_cuda.txt"

    @property
    def name(self) -> str:
        if self.method == "cpu":
            return "cpu"
        elif self.method == "straus":
            return "straus-%d-%d" % (self.C, self.R)
        else:
            return "pippenger-%d" % (self.C)

    @property
    def logs_dir(self) -> str:
        return self.name+"_log"

    def execute_straus(self, curve, i):
        cmd = "%s %s compute params-%d input-%d output-%d straus preprocessed-%d" % (
            cudaprover_path, curve, i, i, i, i)
        return call_foroutput(cmd, output_file=os.path.join(self.logs_dir, "stdout-%d.txt" % i))

    def execute_cpu(self, curve, i):
        cmd = "%s %s compute params-%d input-%d output-%d" % (
            main_path, curve, i, i, i)
        return call_foroutput(cmd, output_file=os.path.join(self.logs_dir, "stdout-%d.txt" % i))

    def execute_pippenger(self, curve, i):
        cmd = "%s %s compute params-%d input-%d output-%d pippenger %d" % (
            cudaprover_path, curve, i, i, i, self.C)
        return call_foroutput(cmd, output_file=os.path.join(self.logs_dir, "stdout-%d.txt" % i))

    def execute(self, curve, i):
        if self.method == "straus":
            return self.execute_straus(curve, i)
        elif self.method == "pippenger":
            return self.execute_pippenger(curve, i)
        elif self.method == "cpu":
            return self.execute_cpu(curve, i)
        else:
            os.abort()


class RunningItem:
    def __init__(self, data: DataConfig, method: MethodConfig) -> None:
        self.data = data
        self.method = method
        self._stdouts: List[str] = []
        self.profile: pd.DataFrame = None

    @property
    def name(self) -> str:
        return self.method.name

    @property
    def able(self) -> bool:
        if self.method.method == "straus":
            return self.data.preprocessed_able
        else:
            return self.data.able

    def prepare(self) -> None:
        if not self.data.able:
            self.data.generate()
        if self.method.method == "straus" and not self.data.preprocessed_able:
            self.data.preprocess()
        assert self.able

    def checksum_right(self) -> bool:
        if os.path.exists("checksum.txt"):
            checksum1 = call_foroutput("sha256sum %s" %
                                       self.method.checksum_filepath).split(' ')[0]
            checksum2 = call_foroutput("sha256sum checksum.txt").split(' ')[0]
            assert checksum1 == checksum2
        else:
            print("Std Checksum Not Found, Result Not Checked")

    def run(self) -> None:
        self.prepare()
        os.chdir(self.data.work_dir)
        os.system("rm output*")
        if os.path.exists(self.method.logs_dir):
            shutil.rmtree(self.method.logs_dir)
        os.mkdir(self.method.logs_dir)
        with open(self.method.checksum_filepath, "w") as f:
            for i in range(0, self.data.test_times):
                with open(os.path.join(self.method.logs_dir, "running-log.txt"), "w") as log:
                    t = time.time()
                    out = self.method.execute(self.data.curve, i)
                    self._stdouts.append(out)
                    log.write("%f\n" % (time.time()-t))
                    checksum = call_foroutput(
                        "sha256sum output-%d" % i).split(' ')[0]
                    f.write(checksum)
                    f.write("\n")
                print("Sleep 5s ...")
                time.sleep(5)
        self.checksum_right()
        os.chdir(base_dir)

    def read_output(self) -> None:
        os.chdir(self.data.work_dir)
        for i in range(0, self.data.test_times):
            with open(os.path.join(self.method.logs_dir, "stdout-%d.txt" % i), "r") as f:
                out = f.read()
                self._stdouts.append(out)
        os.chdir(base_dir)

    def parse_output(self) -> None:
        profiles: List[pd.DataFrame] = []
        for i in range(0, len(self._stdouts)):
            profiles.append(
                Profile(self.data.iname(i), self.method.name, self._stdouts[i]).base_data)
        self.profile = pd.concat(profiles)


class TestSuit:
    def __init__(self) -> None:
        dataconfigs = [
            #DataConfig("MNT6753", 15, 1),
            DataConfig("MNT4753", 15, 1),
        ]
        methods = [
            #MethodConfig("cpu"),
            MethodConfig("straus", 5, 32),
            #MethodConfig("pippenger", 6),
            MethodConfig("pippenger", 7),
            #MethodConfig("pippenger", 8),
            #MethodConfig("pippenger", 9),
            #MethodConfig("pippenger", 10)
        ]
        self.items: List[RunningItem] = []
        for d in dataconfigs:
            for m in methods:
                #pass
                self.items.append(RunningItem(d, m))
        dataconfigs = [
            DataConfig("MNT4753", 18, 1)
        ]
        methods = [
            MethodConfig("cpu"),
            MethodConfig("pippenger", 6),
            MethodConfig("pippenger", 7),
            MethodConfig("pippenger", 8),
            MethodConfig("pippenger", 9),
            MethodConfig("pippenger", 10)
        ]
        for d in dataconfigs:
            for m in methods:
                pass
                #self.items.append(RunningItem(d, m))

    def run(self) -> None:
        for it in self.items:
            it.run()
        self.profile_only()

    def profile_only(self) -> None:
        profiles: List[pd.DataFrame] = []
        for it in self.items:
            it.read_output()
            it.parse_output()
            profiles.append(it.profile)
        profile = pd.concat(profiles)
        filename = "%s.xlsx" % (time.strftime("%m%d%H%M%S"))
        profile.to_excel(os.path.join(report_dir, filename))


if __name__ == "__main__":
    suit = TestSuit()
    suit.run()
    #suit.profile_only()
