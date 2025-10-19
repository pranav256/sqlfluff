use std::vec;
use std::{
    fmt::Display,
    hash::{Hash, Hasher},
};

use crate::{
    dialect::Dialect,
    parser_cache::{CacheKey, ParseCache},
    token::Token,
    parser::{ParseFrame, FrameState, FrameContext, BracketedState, DelimitedState},
};

// Re-export types from the new parser module for backward compatibility
pub use crate::parser::{Grammar, ParseMode, Node, ParseError, SegmentDef, Parsed, ParseErrorType, ParseContext};

// Import utility functions
use crate::parser::utils::{
    tag_keyword_if_word, is_grammar_optional, apply_parse_mode_to_result
};

// Import macros
use crate::find_longest_match;

pub struct Parser<'a> {
    pub tokens: &'a [Token],
    pub pos: usize, // current position in tokens
    pub dialect: Dialect,
    pub parse_cache: ParseCache,
    pub collected_transparent_positions: std::collections::HashSet<usize>, // Track which token positions have had transparent tokens collected
    pub use_iterative_parser: bool, // Full iterative parser (experimental)
}

impl<'a> Parser<'a> {
    pub fn parse_with_grammar_cached(
        &mut self,
        grammar: &Grammar,
        parent_terminators: &[Grammar],
    ) -> Result<Node, ParseError> {
        if self.use_iterative_parser {
            self.parse_with_grammar_cached_iterative(grammar, parent_terminators)
        } else {
            self.parse_with_grammar_cached_recursive(grammar, parent_terminators)
        }
    }

    /// Iterative (frame-based) parser with caching.
    /// Uses a stack-based approach to avoid deep recursion.
    fn parse_with_grammar_cached_iterative(
        &mut self,
        grammar: &Grammar,
        parent_terminators: &[Grammar],
    ) -> Result<Node, ParseError> {
        self.parse_iterative(grammar, parent_terminators)
    }

    /// Recursive parser with caching.
    /// Uses traditional recursive descent with memoization for performance.
    fn parse_with_grammar_cached_recursive(
        &mut self,
        grammar: &Grammar,
        parent_terminators: &[Grammar],
    ) -> Result<Node, ParseError> {
        // Create cache key
        let cache_key = CacheKey::new(self.pos, grammar, self.tokens);

        // Check cache first
        if let Some(Ok((node, end_pos, collected_positions))) = self.parse_cache.get(&cache_key) {
            self.pos = end_pos; // Directly set to cached end position
                                // Restore collected transparent positions
            let num_positions = collected_positions.len();
            for pos in collected_positions {
                self.collected_transparent_positions.insert(pos);
            }
            log::debug!("Cache HIT restored {} collected positions", num_positions);
            return Ok(node);
        }

        // Cache miss - parse fresh
        let positions_before = self.collected_transparent_positions.clone();
        let result = self.parse_with_grammar(grammar, parent_terminators);

        // Store in cache with collected positions
        if let Ok(ref node) = result {
            // Collect positions that were added during this parse
            let new_positions: Vec<usize> = self
                .collected_transparent_positions
                .difference(&positions_before)
                .copied()
                .collect();

            self.parse_cache
                .put(cache_key, Ok((node.clone(), self.pos, new_positions)));
        }

        result
    }

    /// Prune options based on simple matchers before attempting full parse.
    ///
    /// This is the Rust equivalent of Python's `prune_options()` function.
    /// It filters the list of grammar options to only those that could possibly
    /// match the current token, based on quick checks of raw strings and types.
    pub(crate) fn prune_options<'g>(&self, options: &'g [Grammar]) -> Vec<&'g Grammar> {
        // Find first code (non-whitespace) token from current position
        let first_code_token = self.tokens.iter().skip(self.pos).find(|t| t.is_code());

        // If no code token found, can't prune - return all options
        let Some(first_token) = first_code_token else {
            return options.iter().collect();
        };

        // Get token properties for matching
        let first_raw = first_token.raw().to_uppercase();
        let first_type = first_token.get_type();

        log::debug!(
            "Pruning {} options at pos {} (token: '{}', type: {})",
            options.len(),
            self.pos,
            first_raw,
            first_type
        );

        let mut pruned = Vec::new();

        for opt in options {
            // Try to get simple representation
            match opt.simple() {
                None => {
                    // Complex grammar - must try full match
                    log::debug!("  Keeping complex grammar: {}", opt);
                    pruned.push(opt);
                }
                Some(simple) => {
                    // Check if simple matches current token
                    if simple.could_match(first_token) {
                        log::debug!("  Keeping matched grammar: {}", opt);
                        pruned.push(opt);
                    } else {
                        log::debug!("  PRUNED grammar: {}", opt);
                    }
                }
            }
        }

        log::debug!(
            "Pruned from {} to {} options ({:.1}% reduction)",
            options.len(),
            pruned.len(),
            100.0 * (1.0 - pruned.len() as f64 / options.len() as f64)
        );

        pruned
    }

    /// Print cache statistics
    pub fn print_cache_stats(&self) {
        let (hits, misses, hit_rate) = self.parse_cache.stats();
        eprintln!("Parse Cache Statistics:");
        eprintln!("  Hits: {}", hits);
        eprintln!("  Misses: {}", misses);
        eprintln!("  Hit Rate: {:.2}%", hit_rate * 100.0);
    }

    pub fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.pos)
    }

    pub fn bump(&mut self) {
        self.pos += 1;
    }

    pub fn is_at_end(&self) -> bool {
        self.pos >= self.tokens.len()
    }

    /// Collect all transparent tokens (whitespace, newlines) as nodes
    pub fn collect_transparent(&mut self, allow_gaps: bool) -> Vec<Node> {
        let mut transparent_nodes = Vec::new();

        if !allow_gaps {
            return transparent_nodes;
        }

        while let Some(tok) = self.peek() {
            if tok.is_code() {
                break;
            }

            let token_pos = self.pos;

            // Skip if already collected
            if self.collected_transparent_positions.contains(&token_pos) {
                self.bump();
                continue;
            }

            let tok_type = tok.get_type();
            let node = if tok_type == "whitespace" {
                Node::Whitespace(tok.raw(), token_pos)
            } else if tok_type == "newline" {
                Node::Newline(tok.raw(), token_pos)
            } else if tok_type == "end_of_file" {
                Node::EndOfFile(tok.raw(), token_pos)
            } else {
                Node::Code(tok.raw(), token_pos) // Fallback for other non-code tokens
            };

            log::debug!(
                "TRANSPARENT collecting token at pos {}: {:?}",
                token_pos,
                tok
            );
            transparent_nodes.push(node);
            self.collected_transparent_positions.insert(token_pos);
            self.bump();
        }

        transparent_nodes
    }

    /// Skip all transparent tokens (whitespace, newlines) without collecting them
    pub fn skip_transparent(&mut self, allow_gaps: bool) {
        if !allow_gaps {
            return;
        }
        while let Some(tok) = self.peek() {
            match tok {
                tok if !tok.is_code() => {
                    log::debug!("NOCODE skipping token: {:?}", tok);
                    self.bump()
                }
                _ => break,
            }
        }
    }

    /// Move an index forward through tokens until tokens[index] is code.
    /// Returns the index of the first code token, or max_idx if none found.
    pub(crate) fn skip_start_index_forward_to_code(&self, start_idx: usize, max_idx: usize) -> usize {
        for _idx in start_idx..max_idx {
            if self.tokens[_idx].is_code() {
                return _idx;
            }
        }
        max_idx
    }

    /// Move an index backward through tokens until tokens[index - 1] is code.
    /// Returns the index after the last code token, or min_idx if none found.
    fn skip_stop_index_backward_to_code(&self, stop_idx: usize, min_idx: usize) -> usize {
        for _idx in (min_idx + 1..=stop_idx).rev() {
            if self.tokens[_idx - 1].is_code() {
                return _idx;
            }
        }
        min_idx
    }

    /// Trim forward segments based on terminators.
    ///
    /// Given a forward set of segments, find the first terminator and return
    /// the index to use as max_idx (trimmed to last code segment before terminator).
    ///
    /// If no terminators are found, returns the original tokens.len().
    pub(crate) fn trim_to_terminator(&mut self, start_idx: usize, terminators: &[Grammar]) -> usize {
        if start_idx >= self.tokens.len() {
            return self.tokens.len();
        }

        let saved_pos = self.pos;
        self.pos = start_idx;

        // Check if already at a terminator immediately
        if self.is_terminated(terminators) {
            self.pos = saved_pos;
            return start_idx;
        }

        // Find first terminator position
        let mut term_pos = self.tokens.len();
        for idx in start_idx..self.tokens.len() {
            self.pos = idx;
            if self.is_terminated(terminators) {
                term_pos = idx;
                break;
            }
        }

        self.pos = saved_pos;

        // Skip backward from terminator to last code
        self.skip_stop_index_backward_to_code(term_pos, start_idx)
    }

    pub fn is_terminated(&mut self, terminators: &[Grammar]) -> bool {
        let init_pos = self.pos;
        self.skip_transparent(true);
        let saved_pos = self.pos;

        // Check if we've reached end of file
        if self.is_at_end() {
            log::debug!("  TERMED Reached end of file");
            self.pos = init_pos; // restore to position before skipping transparent
            return true;
        }

        // Check if current token is end_of_file type
        if let Some(tok) = self.peek() {
            if tok.get_type() == "end_of_file" {
                log::debug!("  TERMED Found end_of_file token");
                self.pos = init_pos; // restore to position before skipping transparent
                return true;
            }
        }

        log::debug!(
            "  TERM Checking terminators: {:?} at pos {:?}",
            terminators,
            self.pos
        );

        // Temporarily disable iterative parser when checking terminators
        // to avoid nested iterative parses with conflicting frame IDs
        let was_iterative = self.use_iterative_parser;
        self.use_iterative_parser = false;

        for term in terminators {
            if let Ok(node) = self.parse_with_grammar_cached(term, &[]) {
                self.pos = saved_pos; // don't consume

                // Check if the node is "empty" in various ways
                let is_empty = node.is_empty();

                if !is_empty {
                    log::debug!("  TERMED Terminator matched: {}", term);
                    self.pos = init_pos; // restore original position
                    self.use_iterative_parser = was_iterative; // restore flag
                    return true;
                }
            }
            self.pos = saved_pos;
        }

        self.use_iterative_parser = was_iterative; // restore flag
        log::debug!("  NOTERM No terminators matched");
        self.pos = init_pos; // restore original position
        false
    }

    /// Get the grammar for a rule by name.
    /// This is used by the iterative parser to expand Ref nodes into their grammars.
    pub fn get_rule_grammar(&self, name: &str) -> Result<Grammar, ParseError> {
        // Look up the grammar for the segment
        match self.get_segment_grammar(name) {
            Some(g) => Ok(g.clone()),
            None => Err(ParseError::unknown_segment(name.to_string())),
        }
    }

    /// Call a grammar rule by name, producing a Node.
    pub fn call_rule(
        &mut self,
        name: &str,
        parent_terminators: &[Grammar],
    ) -> Result<Node, ParseError> {
        // Look up the grammar for the segment
        let grammar = match self.get_segment_grammar(name) {
            Some(g) => g,
            None => return Err(ParseError::unknown_segment(name.to_string())),
        };

        // Parse using the grammar
        let node = self.parse_with_grammar_cached(grammar, parent_terminators)?;

        // If the node is empty, return it as-is without wrapping
        // This prevents infinite loops when optional segments match nothing
        if node.is_empty() {
            return Ok(node);
        }

        // Get the segment type from the dialect
        let segment_type = self.dialect.get_segment_type(name).map(|s| s.to_string());

        // Check if this is a KeywordSegment and the child is a word token
        // If so, convert Node::Code to Node::Keyword
        let processed_node = if name.ends_with("KeywordSegment") {
            tag_keyword_if_word(&node, &self.tokens)
        } else {
            node
        };

        // Wrap in a Ref node for type clarity
        Ok(Node::Ref {
            name: name.to_string(),
            segment_type,
            child: Box::new(processed_node),
        })
    }

    /// Lookup SegmentDef by name
    pub fn get_segment_grammar(&self, name: &str) -> Option<&'static Grammar> {
        self.dialect.get_segment_grammar(name)
    }

    pub fn new(tokens: &'a [Token], dialect: Dialect) -> Parser<'a> {
        Parser {
            tokens,
            pos: 0,
            dialect,
            parse_cache: ParseCache::new(),
            collected_transparent_positions: std::collections::HashSet::new(),
            use_iterative_parser: true, // Default to iterative parser
        }
    }
}

// ParseContext, ParseError, Parsed, and ParseErrorType have been moved to src/parser/types.rs
