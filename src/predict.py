import logging
import os
import re

import mlflow
import numpy as np
import pandas as pd
import torch
import json
from onnxruntime import InferenceSession
from onnxruntime.capi.onnxruntime_inference_collection import Session
from sqlalchemy import VARCHAR, create_engine
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from typing import Any

from ******** import CustomDataset
from ******** import crt_preprocessing, \
    note_preprocessing
from ******** import BERT_LABELS

logger = logging.getLogger(__name__)

os.environ['MLFLOW_TRACKING_URI'] = 'https://mlflow-k8s-app.query.consul-tech/'


class EmptyDataFrameError(Exception):
    """No new data in the dataframe."""


class DataFrameAlreadyExistError(Exception):
    """No new data in the dataframe."""


class Predictor:

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize parameters of the Predictor class.

        Args:

            mlflow_model: The model name in mlflow.

            dryrun: If False, the results of the model will not be saved in the database.

            target_table: Table from which we extract the chat texts for tags from tag_ids.

            batch_size: Model's batch size.

            max_length: Maximum number of tokens loaded into the model.

            conn: Connection for sql engine.

            tokenizer_path: Path to the tokenizer vocab and json files.

            layer: Stage, Production, None.

            thresholds: Tag thresholds

        """

        self.tokenizer = None
        self.mlflow_model = config["mlflow_model"]
        self.dryrun = config["dryrun"]
        self.target_table = config["target_table"]
        self.batch_size = config["batch_size"]
        self.max_length = config["max_length"]
        self.dwh_conn = create_engine(config["conn"])
        self.tokenizer_path = config["tokenizer_path"]
        self.layer = config["layer"]
        self.thresholds = config["thresholds"]

    def step_predict(self) -> None:
        """We retrieve call data from dwh."""

        logger.info('step_predict: init')

        query = """
                    **********
                    **********
                    **********
                    **********
                    **********
                    **********
                    **********
                """

        df_call = pd.read_sql(query, self.dwh_conn)

        if df_call.empty:
            logger.error('Empty dataframe')
            raise DataFrameAlreadyExistError()

        logger.info('download tokenizer from mlflow: start')
        mlflow.artifacts.download_artifacts(
            run_id='********',
            dst_path=self.tokenizer_path
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        logger.info('download tokenizer from mlflow: complete')

        self.predict(df_call)

    def predict(self, dataframe: pd.DataFrame) -> None:
        """Predicting tags for calls".

        Args:
            dataframe (pd.DataFrame): Call dataset.

        """
        model_session = self.download_model()
        updated_dataframe, dataloader = self.preprocess(dataframe)
        predictions = self.predict_with_model(model_session, dataloader)
        self.save_dataframe(
            predictions=predictions,
            dataframe=updated_dataframe,
            table=self.target_table,
            mode='append',
        )

    def download_model(self) -> Session:
        """Downloading mlflow model".

        Returns:
            Session: Session for onnx mlflow model.

        """

        logger.info('download model from mlflow: start')
        model_path = f'models:/{self.mlflow_model}/{self.layer}'
        onnx_model = mlflow.onnx.load_model(model_path)
        onnx_session_mlflow = InferenceSession(onnx_model.SerializeToString())
        logger.info('download model from mlflow: complete')
        return onnx_session_mlflow

    def preprocess(self, df: pd.DataFrame) -> tuple[pd.DataFrame, DataLoader[CustomDataset]]:
        """Text pre-processing and dataloader creation for ml-model.

        Args:
            df (pd.DataFrame): Dataframe for pre-processing.

        Returns (tuple[pd.DataFrame, DataLoader[CustomDataset]]): We will return the updated
                                                                  dataframe and dataloader.

        """
        logger.info('Preprocessing: start')

        df = df.drop_duplicates(subset=["****"], keep="last")
        df["****"] = df["****"].apply(crt_preprocessing)
        df["****"] = df["****"].apply(note_preprocessing)
        df.dropna(inplace=True)
        df = df.reset_index(drop=True)

        if df.empty:
            logger.error('Empty dataframe')
            raise EmptyDataFrameError()

        dataset = CustomDataset(df.body_str.values, df.result.values,
                                self.tokenizer, self.max_length)

        loader_params = {
            'batch_size': self.batch_size,
            'shuffle': False,
            'num_workers': 0,
        }
        dataloader = DataLoader(dataset, **loader_params)

        logger.info('Preprocessing: complete')
        return df, dataloader

    def predict_with_model(
        self,
        model_session: Session,
        dataloader: DataLoader[CustomDataset],
    ) -> tuple[list[float], list[str]]:
        """Getting model predictions.

        Args:
            model_session (Session): Onnx session for scoring data.
            dataloader (DataLoader[CustomDataset]): Dataloader for current "model_session"

        Returns:
            tuple[list[float],list[str]]: Model predictions.
        """
        logger.info('predict: start')

        prediction_list = []
        for batch in dataloader:
            with torch.no_grad():
                input_feed = {
                    "input_ids_note": batch['input_ids_note'].int().cpu().numpy(),
                    "attention_mask_note": batch['attention_mask_note'].int().cpu().numpy(),
                    "input_ids_content": batch['input_ids_content'].int().cpu().numpy(),
                    "attention_mask_content": batch['attention_mask_content'].int().cpu().numpy(),
                }
                outputs = model_session.run(['output'], input_feed)[0]
                prediction_list.extend(
                    torch.sigmoid(torch.tensor(outputs)).cpu().detach().numpy()
                )

        probs = prediction_list.copy()

        prediction_list = np.array(prediction_list)

        for i, threshold in enumerate(self.thresholds):
            prediction_list[:, i] = (prediction_list[:, i] >= threshold).astype(int)

        other_tag = np.all(prediction_list == 0, axis=1)

        results_json_targets = []
        results_json_probs = []

        for i, item in enumerate(prediction_list):
            results_json_targets.append(json.dumps({
                "targets": dict(zip(BERT_LABELS, np.append(item, int(other_tag[i]))))},
                ensure_ascii=False))

            results_json_probs.append(json.dumps({
                "probs": dict(zip(BERT_LABELS, [str(num) for num in probs[i]]))},
                ensure_ascii=False))

        logger.info('predict: complete')

        return results_json_targets, results_json_probs

    def save_dataframe(self, predictions: tuple[list[float], list[str]], dataframe: pd.DataFrame,
                       table: str, mode: str) -> None:
        """Saving dataframe to dwh.

        Args:
             predictions (tuple[list[float],list[str]]): Data for "json_output_targets" and "json_output_probs" columns.
             dataframe (pd.DataFrame): Final dataframe.
             table (str): Table in which the data are loaded.
             mode (str): if_exists mode.

        """

        if self.dryrun:
            logger.info('save dataframe: skipped')
            return

        logger.info('save dataframe: start')
        dataframe['json_output_targets'] = predictions[0]
        dataframe['json_output_probs'] = predictions[1]
        dataframe[['*****', 'json_output_probs', 'json_output_targets']].to_sql(
            name=table,
            con=self.dwh_conn,
            if_exists=mode,
            index=False,
            dtype={
                'ucid': VARCHAR(36),
                'json_output_probs': VARCHAR(2000),
                'json_output_targets': VARCHAR(2000),

            },
            chunksize=1000
        )
        logger.info('save dataframe: complete')
