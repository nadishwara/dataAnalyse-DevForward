# DevForward: Impostor Phenomenon (IP) Linguistic Annotation Guidelines (v1.1)

**Project:** DevForward — NLP-Based Identification of Impostor-Related Linguistic Patterns in Software Developer Communications

**Theoretical Foundation:** CIPS-informed linguistic annotation framework

**Annotation Scope:** Non-diagnostic identification of explicitly expressed potential impostor-related linguistic patterns.

## 1. Research Task Definition

The objective is to identify explicit linguistic expressions potentially associated with the Impostor Phenomenon in software developer communications and reflections.

The annotation task distinguishes impostor-related expressions from ordinary negative sentiment, technical frustration, realistic assessments of limited experience, and general professional self-doubt.

Annotations describe evidence within a text rather than diagnose the author's psychological condition.

The annotation categories are informed by Impostor Phenomenon literature and require task-specific validation.

## 2. Unit of Annotation

- One complete developer text record associated with a unique Source_ID.
- Evaluate only evidence available within the supplied text.
- Do not infer unstated psychological beliefs or personal history.
- Preserve original source metadata and sentiment labels.
- Maintain separate LLM-suggested and human-reviewed annotations.

## 3. Hierarchical Decision Workflow

### Stage A — Relevance Screening

**Relevant:** The text contains meaningful personal appraisal of professional competence, achievement, recognition, or perceived ability.

**Not_relevant:** The text primarily contains technical information, general frustration, factual comparisons, or other content without meaningful professional self-appraisal.

**Uncertain:** The available context is insufficient to determine relevance.

### Stage B — Potential IP Pattern Classification

**Present:** Explicit textual evidence supports at least one defined impostor-related linguistic pattern.

**Absent:** No defined impostor-related pattern is supported by the available text.

**Uncertain:** The expression is ambiguous or insufficiently contextualised to make a reliable decision.

Professional self-appraisal may be Relevant while the IP pattern remains Absent.

## 4. Pattern Categories and Decision Rules

### 4.1 Fear_of_Incompetence

**Include:** Explicit concern about being exposed as less competent, capable, or knowledgeable than others believe.

**Exclude:** Ordinary technical insecurity, limited experience, negative skill assessments, and requests for assistance without an explicit impostor-related belief.

**Positive example:** “My colleagues will eventually discover that I am not as competent as they think.”

**Exclusion example:** “I am terrible with C.”

### 4.2 Luck_Attribution

**Include:** Explicit attribution of personal professional success to luck or chance in a way that dismisses personal contribution.

**Exclude:** Ordinary references to luck, uncertainty, or random technical events.

**Positive example:** “I only received the promotion because I was lucky.”

### 4.3 Achievement_Discounting

**Include:** Explicit dismissal, minimisation, or rejection of personal achievements or deserved professional recognition.

**Exclude:** Realistic assessments of task difficulty, code quality, or technical performance.

**Positive example:** “My team praised my contribution, but I do not think I deserve the recognition.”

### 4.4 Other_or_Mixed

Use for multiple explicitly supported patterns, or another pattern that has been separately defined and approved in the annotation protocol.

Do not use this category as a general label for negative sentiment or ambiguous self-doubt.

### 4.5 None

Use when no defined impostor-related linguistic pattern is supported.

### 4.6 Uncertain

Use when ambiguity or missing context prevents a confident annotation.

## 5. Evidence and Annotation Consistency

- Present annotations require a verbatim supporting evidence span from the original text.
- The evidence must semantically support the assigned category.
- Absent annotations must use the category None and an empty IP evidence field.
- Uncertain annotations must use the category Uncertain, with a note explaining the ambiguity.
- Not_relevant records must not receive a Present label.
- Do not infer additional fears, beliefs, or psychological conditions that are not expressed in the text.

## 6. Human Review and Validation

LLM-generated annotations are preliminary suggestions rather than ground-truth labels.

Human reviewers must evaluate the original text against the written guidelines.

Where feasible, independent annotators should review the data without seeing previous LLM suggestions or other reviewers' decisions.

Record individual annotations before resolving disagreements. Preserve original decisions and document the reasons for any changes.

Agreement statistics obtained from reviewing pre-filled LLM annotations must be reported as assisted-review agreement, not independent human validation.

## 7. Version History

**Version 1.0:** Initial annotation framework used during the Senti4SD pilot.

**Version 1.1:** Refined operational definitions, clarified the distinction between general self-criticism and impostor-related expressions, strengthened evidence requirements, and introduced clearer human-validation procedures.

Version 1.1 is a proposed revision requiring review before further formal annotation. Previous pilot results must remain available for reproducibility.
