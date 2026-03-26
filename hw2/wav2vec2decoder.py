import math
from typing import List, Tuple, Union
import heapq

import kenlm
import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC


# ---------------------------------------------------------------------------
# Provided utility — do NOT modify
# ---------------------------------------------------------------------------
def _log_add(a: float, b: float) -> float:
    """Numerically stable log(exp(a) + exp(b))."""
    if a == float('-inf'):
        return b
    if b == float('-inf'):
        return a
    if a > b:
        return a + math.log1p(math.exp(b - a))
    return b + math.log1p(math.exp(a - b))


class Wav2Vec2Decoder:
    def __init__(
            self,
            model_name="facebook/wav2vec2-base-100h",
            lm_model_path="lm/3-gram.pruned.1e-7.arpa.gz",
            beam_width=3,
            alpha=1.0,
            beta=1.0,
            temperature=1.0,
        ):
        """
        Args:
            model_name (str): Pretrained Wav2Vec2 model from HuggingFace.
            lm_model_path (str): Path to a KenLM .arpa/.arpa.gz model.
                Pass None to disable LM (Tasks 1–3).
            beam_width (int): Number of hypotheses kept during beam search.
            alpha (float): LM weight used in shallow fusion and rescoring.
                score = log_p_acoustic + alpha * log_p_lm + beta * num_words
            beta (float): Word insertion bonus (see above).
            temperature (float): Scales acoustic logits before softmax.
                T < 1 sharpens the distribution (model more confident).
                T > 1 flattens it (model less confident, giving LM more
                influence). T = 1.0 leaves logits unchanged.
        """
        # Interact with processor/model ONLY here and in decode() to obtain
        # logits — no further model calls are allowed anywhere else.
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name)

        self.vocab = {i: c for c, i in self.processor.tokenizer.get_vocab().items()}
        self.blank_token_id = self.processor.tokenizer.pad_token_id
        self.word_delimiter = self.processor.tokenizer.word_delimiter_token

        self.beam_width = beam_width
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature

        self.lm_model = kenlm.Model(lm_model_path) if lm_model_path else None

    # -----------------------------------------------------------------------
    # Provided utility — do NOT modify
    # -----------------------------------------------------------------------
    def _ids_to_text(self, token_ids: List[int]) -> str:
        """Convert a list of token IDs to a decoded string."""
        text = ''.join(self.vocab[i] for i in token_ids)
        return text.replace(self.word_delimiter, ' ').strip().lower()

    # -----------------------------------------------------------------------
    # Tasks 1–4: implement the methods below
    # -----------------------------------------------------------------------

    def greedy_decode(self, logits: torch.Tensor) -> str:
        """
        Perform greedy decoding (find best CTC path).

        Args:
            logits (torch.Tensor): Logits from Wav2Vec2 model (T, V).

        Returns:
            str: Decoded transcript.
        """
        # <YOUR CODE GOES HERE>
        log_probs = torch.log_softmax(logits, dim=-1)
        best_ids = torch.argmax(log_probs, dim=-1).tolist()

        collapsed = []
        prev = None
        for idx in best_ids:
            if idx != prev:
                if idx != self.blank_token_id:
                    collapsed.append(idx)
                prev = idx

        return self._ids_to_text(collapsed)

    def beam_search_decode(self, logits: torch.Tensor, return_beams: bool = False):
        """
        Perform CTC prefix beam search decoding (no LM).

        Args:
            logits (torch.Tensor): Logits from Wav2Vec2 model (T, V).
            return_beams (bool): Return all beam hypotheses for second-pass LM rescoring.

        Returns:
            Union[str, List[Tuple[List[int], float]]]:
                str - best decoded transcript (if return_beams=False).
                List[Tuple[List[int], float]] - list of (token_ids, log_prob)
                    tuples sorted best-first (if return_beams=True).
        """
        # <YOUR CODE GOES HERE>
        log_probs = torch.log_softmax(logits, dim=-1)
        T, V = log_probs.shape
        NEG_INF = float('-inf')

        beams = {(): (0.0, NEG_INF)}

        for t in range(T):
            new_beams = {}

            def _merge(prefix, pb, pnb):
                if prefix in new_beams:
                    old_pb, old_pnb = new_beams[prefix]
                    new_beams[prefix] = (_log_add(old_pb, pb), _log_add(old_pnb, pnb))
                else:
                    new_beams[prefix] = (pb, pnb)

            pruned = sorted(beams.items(), key=lambda x: _log_add(x[1][0], x[1][1]), reverse=True)
            pruned = pruned[:self.beam_width]

            for prefix, (pb, pnb) in pruned:
                total = _log_add(pb, pnb)
                lp_blank = log_probs[t, self.blank_token_id].item()

                _merge(prefix, total + lp_blank, NEG_INF)

                for c in range(V):
                    if c == self.blank_token_id:
                        continue
                    lp_c = log_probs[t, c].item()

                    if len(prefix) > 0 and prefix[-1] == c:
                        _merge(prefix + (c,), NEG_INF, pb + lp_c)
                        _merge(prefix, NEG_INF, pnb + lp_c)
                    else:
                        _merge(prefix + (c,), NEG_INF, total + lp_c)

            beams = new_beams

        results = []
        for prefix, (pb, pnb) in beams.items():
            total = _log_add(pb, pnb)
            results.append((list(prefix), total))
        results.sort(key=lambda x: x[1], reverse=True)

        if return_beams:
            return results[:self.beam_width * 3]

        if not results:
            return ""
        return self._ids_to_text(results[0][0])

    def beam_search_with_lm(self, logits: torch.Tensor) -> str:
        """
        Perform beam search decoding with shallow LM fusion.

        Uses the scoring formula:
            score = log_p_acoustic + alpha * log_p_lm + beta * num_words

        LM score is applied at word boundaries (when '|' token appears).
        """
        if not self.lm_model:
            raise ValueError("KenLM model required for LM shallow fusion")
        # <YOUR CODE GOES HERE>
        log_probs = torch.log_softmax(logits, dim=-1)
        T, V = log_probs.shape
        NEG_INF = float('-inf')

        word_delimiter_id = None
        for idx, char in self.vocab.items():
            if char == self.word_delimiter:
                word_delimiter_id = idx
                break

        beams = {(): (0.0, NEG_INF, 0.0, 0)}

        for t in range(T):
            new_beams = {}

            scored = []
            for prefix, (pb, pnb, lm_sc, nw) in beams.items():
                acoustic_total = _log_add(pb, pnb)
                combined = acoustic_total + self.alpha * lm_sc + self.beta * nw
                scored.append((combined, prefix, pb, pnb, lm_sc, nw))

            scored.sort(key=lambda x: x[0], reverse=True)
            scored = scored[:self.beam_width]

            def _update(prefix, new_pb, new_pnb, lm_sc, nw):
                if prefix in new_beams:
                    old_pb, old_pnb, old_lm, old_nw = new_beams[prefix]
                    new_beams[prefix] = (
                        _log_add(old_pb, new_pb),
                        _log_add(old_pnb, new_pnb),
                        max(old_lm, lm_sc),
                        max(old_nw, nw),
                    )
                else:
                    new_beams[prefix] = (new_pb, new_pnb, lm_sc, nw)

            for _, prefix, pb, pnb, lm_sc, nw in scored:
                total_prev = _log_add(pb, pnb)

                lp_blank = log_probs[t, self.blank_token_id].item()
                _update(prefix, total_prev + lp_blank, NEG_INF, lm_sc, nw)

                for c in range(V):
                    if c == self.blank_token_id:
                        continue
                    lp_c = log_probs[t, c].item()

                    if len(prefix) > 0 and prefix[-1] == c:
                        new_prefix = prefix + (c,)
                        _update(new_prefix, NEG_INF, pb + lp_c, lm_sc, nw)
                        _update(prefix, NEG_INF, pnb + lp_c, lm_sc, nw)
                    else:
                        new_prefix = prefix + (c,)
                        new_lm_sc = lm_sc
                        new_nw = nw

                        if c == word_delimiter_id:
                            text = self._ids_to_text(list(new_prefix))
                            if text.strip():
                                lm_score_val = self.lm_model.score(text, bos=True, eos=False)
                                new_lm_sc = lm_score_val * math.log(10)
                                new_nw = len(text.split())

                        _update(new_prefix, NEG_INF, total_prev + lp_c, new_lm_sc, new_nw)

            beams = new_beams

        results = []
        for prefix, (pb, pnb, lm_sc, nw) in beams.items():
            acoustic = _log_add(pb, pnb)
            text = self._ids_to_text(list(prefix))
            if text.strip():
                lm_score_val = self.lm_model.score(text, bos=True, eos=True)
                lm_score_ln = lm_score_val * math.log(10)
                num_words = len(text.split())
            else:
                lm_score_ln = 0.0
                num_words = 0
            combined = acoustic + self.alpha * lm_score_ln + self.beta * num_words
            results.append((list(prefix), combined))

        results.sort(key=lambda x: x[1], reverse=True)

        if not results:
            return ""
        return self._ids_to_text(results[0][0])

    def lm_rescore(self, beams: List[Tuple[List[int], float]]) -> str:
        """
        Perform second-pass LM rescoring on beam search outputs.

        Args:
            beams (List[Tuple[List[int], float]]): List of (token_ids, log_prob)
                tuples from beam_search_decode(logits, return_beams=True).

        Returns:
            str: Best rescored transcript.
        """
        if not self.lm_model:
            raise ValueError("KenLM model required for LM rescoring")
        # <YOUR CODE GOES HERE>
        best_score = float('-inf')
        best_text = ""

        for token_ids, acoustic_score in beams:
            text = self._ids_to_text(token_ids)
            if text.strip():
                lm_score = self.lm_model.score(text, bos=True, eos=True)
                lm_score_ln = lm_score * math.log(10)
                num_words = len(text.split())
            else:
                lm_score_ln = 0.0
                num_words = 0

            combined = acoustic_score + self.alpha * lm_score_ln + self.beta * num_words

            if combined > best_score:
                best_score = combined
                best_text = text

        return best_text

    # -----------------------------------------------------------------------
    # Provided — do NOT modify
    # -----------------------------------------------------------------------
    def decode(self, audio_input: torch.Tensor, method: str = "greedy") -> str:
        """
        Run the full decoding pipeline on a raw audio tensor.

        Args:
            audio_input (torch.Tensor): 1-D or 2-D audio waveform at 16 kHz.
            method (str): One of "greedy", "beam", "beam_lm", "beam_lm_rescore".

        Returns:
            str: Decoded transcript (lowercase).
        """
        inputs = self.processor(audio_input, return_tensors="pt", sampling_rate=16000)
        with torch.no_grad():
            logits = self.model(inputs.input_values.squeeze(0)).logits[0]
        # Temperature scaling (Task 3): flatten/sharpen the distribution
        # before log_softmax.  T=1.0 is a no-op.  Your decoders must call
        # torch.log_softmax on the logits they receive — do not call it here.
        logits = logits / self.temperature

        if method == "greedy":
            return self.greedy_decode(logits)
        elif method == "beam":
            return self.beam_search_decode(logits)
        elif method == "beam_lm":
            return self.beam_search_with_lm(logits)
        elif method == "beam_lm_rescore":
            beams = self.beam_search_decode(logits, return_beams=True)
            return self.lm_rescore(beams)
        else:
            raise ValueError(
                f"Unknown method '{method}'. "
                "Choose one of: 'greedy', 'beam', 'beam_lm', 'beam_lm_rescore'."
            )


# ---------------------------------------------------------------------------
# Quick debug helper — run this file directly to sanity-check your decoder
# on the provided examples/ clips before evaluating on the full test sets.
# ---------------------------------------------------------------------------

def test(decoder: Wav2Vec2Decoder, audio_path: str, reference: str) -> None:
    import jiwer

    audio_input, sr = torchaudio.load(audio_path)
    assert sr == 16000, f"Expected 16 kHz, got {sr} Hz for {audio_path}"

    print("=" * 60)
    print(f"REF : {reference}")

    for method in ["greedy", "beam", "beam_lm", "beam_lm_rescore"]:
        try:
            hyp = decoder.decode(audio_input, method=method)
        except NotImplementedError:
            print(f"  [{method}] not yet implemented")
            continue
        except ValueError as e:
            print(f"  [{method}] skipped ({e})")
            continue
        cer = jiwer.cer(reference, hyp)
        wer = jiwer.wer(reference, hyp)
        print(f"  [{method}] {hyp}")
        print(f"           WER={wer:.2%}  CER={cer:.2%}")


if __name__ == "__main__":
    test_samples = [
        ("examples/sample1.wav", "if you are generous here is a fitting opportunity for the exercise of your magnanimity if you are proud here am i your rival ready to acknowledge myself your debtor for an act of the most noble forbearance"),
        ("examples/sample2.wav", "and if any of the other cops had private rackets of their own izzy was undoubtedly the man to find it out and use the information with a beat such as that even going halves and with all the graft to the upper brackets he'd still be able to make his pile in a matter of months"),
        ("examples/sample3.wav", "guess a man gets used to anything hell maybe i can hire some bums to sit around and whoop it up when the ships come in and bill this as a real old martian den of sin"),
        ("examples/sample4.wav", "it was a tune they had all heard hundreds of times so there was no difficulty in turning out a passable imitation of it to the improvised strains of i didn't want to do it the prisoner strode forth to freedom"),
        ("examples/sample5.wav", "marguerite tired out with this long confession threw herself back on the sofa and to stifle a slight cough put up her handkerchief to her lips and from that to her eyes"),
        ("examples/sample6.wav", "at this time all participants are in a listen only mode"),
        ("examples/sample7.wav", "the increase was mainly attributable to the net increase in the average size of our fleets"),
        ("examples/sample8.wav", "operating surplus is a non cap financial measure which is defined as fully in our press release"),
    ]
    decoder = Wav2Vec2Decoder(
        lm_model_path="lm/3-gram.pruned.1e-7.arpa.gz",
        beam_width=10,
        alpha=0.5,
        beta=1.0,
    )
    for audio_path, reference in test_samples:
        test(decoder, audio_path, reference)