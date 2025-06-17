"""
Symbolic Residue Analyzer: Tools for capturing and analyzing unexplained patterns and phenomena
in complex systems across scales

This module provides tools for systematically documenting and analyzing patterns that don't
fit neatly into current understanding but might contain seeds of deeper insights.

Author: David Kimai
Created: June 16, 2025
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
import uuid
import datetime
import json
import os
from dataclasses import dataclass, field, asdict
import matplotlib.pyplot as plt
from scipy import stats, signal

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class SymbolicResidue:
    """Class for storing and managing symbolic residue observations."""
    
    id: str = field(default_factory=lambda: f"SR{uuid.uuid4().hex[:8]}")
    timestamp: str = field(default_factory=lambda: datetime.datetime.now().isoformat())
    observer: str = "system"
    system_context: str = ""
    title: str = ""
    observation: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    repeatability: str = "unknown"
    documentation: List[str] = field(default_factory=list)
    initial_hypotheses: List[str] = field(default_factory=list)
    relationship_to_known: str = ""
    cross_system_parallels: List[str] = field(default_factory=list)
    potential_significance: str = ""
    status: str = "documented"
    investigation_steps: List[str] = field(default_factory=list)
    open_questions: List[str] = field(default_factory=list)
    next_steps: List[str] = field(default_factory=list)
    priority: str = "medium"
    recursive_nature: str = ""
    meta_implications: str = ""
    researcher_effect: str = ""
    personal_reflection: str = ""
    update_history: List[Dict[str, Any]] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)
    
    def update(self, **kwargs):
        """Update residue with new information."""
        update_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "fields_updated": list(kwargs.keys()),
            "previous_values": {}
        }
        
        for key, value in kwargs.items():
            if hasattr(self, key):
                update_entry["previous_values"][key] = getattr(self, key)
                setattr(self, key, value)
            else:
                logger.warning(f"Unknown field: {key}")
        
        self.update_history.append(update_entry)

class ResidueRepository:
    """Repository for managing symbolic residue observations."""
    
    def __init__(self, save_dir: str = "./residue"):
        """
        Initialize the repository.
        
        Parameters:
        -----------
        save_dir : str, default="./residue"
            Directory for saving residue files
        """
        self.save_dir = save_dir
        self.residues = []
        self._ensure_dir()
    
    def _ensure_dir(self):
        """Ensure save directory exists."""
        os.makedirs(self.save_dir, exist_ok=True)
    
    def add(self, residue: Union[SymbolicResidue, Dict[str, Any]]) -> str:
        """
        Add a residue to the repository.
        
        Parameters:
        -----------
        residue : Union[SymbolicResidue, Dict[str, Any]]
            Residue to add
        
        Returns:
        --------
        str
            ID of the added residue
        """
        if isinstance(residue, dict):
            residue = SymbolicResidue(**residue)
        
        self.residues.append(residue)
        self._save_residue(residue)
        logger.info(f"Added residue {residue.id}: {residue.title}")
        
        return residue.id
    
    def get(self, residue_id: str) -> Optional[SymbolicResidue]:
        """
        Get a residue by ID.
        
        Parameters:
        -----------
        residue_id : str
            ID of the residue
        
        Returns:
        --------
        Optional[SymbolicResidue]
            The residue if found, None otherwise
        """
        for residue in self.residues:
            if residue.id == residue_id:
                return residue
        
        # If not in memory, try loading from file
        try:
            return self._load_residue(residue_id)
        except:
            return None
    
    def update(self, residue_id: str, **kwargs) -> bool:
        """
        Update a residue.
        
        Parameters:
        -----------
        residue_id : str
            ID of the residue
        **kwargs
            Fields to update
        
        Returns:
        --------
        bool
            Whether the update was successful
        """
        residue = self.get(residue_id)
        if not residue:
            logger.warning(f"Residue not found: {residue_id}")
            return False
        
        residue.update(**kwargs)
        self._save_residue(residue)
        
        return True
    
    def search(self, query: Dict[str, Any] = None, tags: List[str] = None, 
              system_context: str = None, priority: str = None,
              status: str = None) -> List[SymbolicResidue]:
        """
        Search for residues matching criteria.
        
        Parameters:
        -----------
        query : Dict[str, Any], optional
            Fields to match
        tags : List[str], optional
            Tags to match (any)
        system_context : str, optional
            System context to match
        priority : str, optional
            Priority to match
        status : str, optional
            Status to match
        
        Returns:
        --------
        List[SymbolicResidue]
            Matching residues
        """
        self._load_all()  # Ensure all residues are loaded
        
        results = self.residues.copy()
        
        # Filter by query
        if query:
            for key, value in query.items():
                results = [r for r in results if hasattr(r, key) and getattr(r, key) == value]
        
        # Filter by tags
        if tags:
            results = [r for r in results if any(tag in r.tags for tag in tags)]
        
        # Filter by system context
        if system_context:
            results = [r for r in results if r.system_context == system_context]
        
        # Filter by priority
        if priority:
            results = [r for r in results if r.priority == priority]
        
        # Filter by status
        if status:
            results = [r for r in results if r.status == status]
        
        return results
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert all residues to a DataFrame.
        
        Returns:
        --------
        pd.DataFrame
            DataFrame containing all residues
        """
        self._load_all()  # Ensure all residues are loaded
        
        # Convert to dicts
        dicts = [r.to_dict() for r in self.residues]
        
        # Handle nested fields
        for d in dicts:
            for k, v in d.items():
                if isinstance(v, list) or isinstance(v, dict):
                    d[k] = json.dumps(v, default=str)
        
        return pd.DataFrame(dicts)
    
    def _save_residue(self, residue: SymbolicResidue):
        """
        Save a residue to file.
        
        Parameters:
        -----------
        residue : SymbolicResidue
            Residue to save
        """
        filename = os.path.join(self.save_dir, f"{residue.id}.json")
        
        with open(filename, 'w') as f:
            f.write(residue.to_json())
    
    def _load_residue(self, residue_id: str) -> SymbolicResidue:
        """
        Load a residue from file.
        
        Parameters:
        -----------
        residue_id : str
            ID of the residue
        
        Returns:
        --------
        SymbolicResidue
            Loaded residue
        
        Raises:
        -------
        FileNotFoundError
            If residue file not found
        """
        filename = os.path.join(self.save_dir, f"{residue_id}.json")
        
        with open(filename, 'r') as f:
            data = json.load(f)
        
        return SymbolicResidue(**data)
    
    def _load_all(self):
        """Load all residues from files."""
        self._ensure_dir()
        
        # Get all residue files
        files = [f for f in os.listdir(self.save_dir) if f.endswith('.json')]
        
        # Get existing residue IDs
        existing_ids = [r.id for r in self.residues]
        
        # Load new residues
        for file in files:
            residue_id = file[:-5]  # Remove .json
            
            if residue_id not in existing_ids:
                try:
                    residue = self._load_residue(residue_id)
                    self.residues.append(residue)
                except Exception as e:
                    logger.warning(f"Failed to load residue {residue_id}: {e}")

class ResidueAnalyzer:
    """Analyzer for detecting patterns in symbolic residue."""
    
    def __init__(self, repository: ResidueRepository):
        """
        Initialize the analyzer.
        
        Parameters:
        -----------
        repository : ResidueRepository
            Repository containing residues to analyze
        """
        self.repository = repository
    
    def find_clusters(self, fields: List[str] = None, n_clusters: int = None,
                     method: str = 'kmeans') -> Dict[str, List[str]]:
        """
        Find clusters of related residues.
        
        Parameters:
        -----------
        fields : List[str], optional
            Fields to use for clustering
        n_clusters : int, optional
            Number of clusters (if None, determined automatically)
        method : str, default='kmeans'
            Clustering method ('kmeans' or 'hierarchical')
        
        Returns:
        --------
        Dict[str, List[str]]
            Clusters of residue IDs
        """
        # Default fields
        if fields is None:
            fields = ['observation', 'potential_significance', 'relationship_to_known']
        
        # Get all residues
        all_residues = self.repository.residues
        
        # Extract text from specified fields
        texts = []
        for residue in all_residues:
            text = ""
            for field in fields:
                if hasattr(residue, field):
                    value = getattr(residue, field)
                    if isinstance(value, str):
                        text += value + " "
                    elif isinstance(value, list) and all(isinstance(x, str) for x in value):
                        text += " ".join(value) + " "
            texts.append(text)
        
        # Skip if not enough text
        if not texts or all(not text.strip() for text in texts):
            return {}
        
        # Create TF-IDF vectors
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.cluster import KMeans, AgglomerativeClustering
        
        vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        try:
            X = vectorizer.fit_transform(texts)
        except Exception as e:
            logger.warning(f"Vectorization failed: {e}")
            return {}
        
        # Skip if not enough features
        if X.shape[1] == 0:
            return {}
        
        # Determine number of clusters if not specified
        if n_clusters is None:
            from sklearn.metrics import silhouette_score
            
            # Try different numbers of clusters
            scores = []
            max_clusters = min(10, len(texts) // 2) if len(texts) > 4 else 2
            for k in range(2, max_clusters + 1):
                if method == 'kmeans':
                    clusterer = KMeans(n_clusters=k, random_state=42)
                else:
                    clusterer = AgglomerativeClustering(n_clusters=k)
                
                labels = clusterer.fit_predict(X.toarray() if method == 'kmeans' else X.toarray())
                
                if len(set(labels)) <= 1:
                    continue
                
                score = silhouette_score(X, labels)
                scores.append((k, score))
            
            if not scores:
                n_clusters = 2
            else:
                # Choose number of clusters with highest score
                n_clusters = max(scores, key=lambda x: x[1])[0]
        
        # Perform clustering
        if method == 'kmeans':
            clusterer = KMeans(n_clusters=n_clusters, random_state=42)
            labels = clusterer.fit_predict(X.toarray())
        else:
            clusterer = AgglomerativeClustering(n_clusters=n_clusters)
            labels = clusterer.fit_predict(X.toarray())
        
        # Group residues by cluster
        clusters = {}
        for i, label in enumerate(labels):
            cluster_id = f"Cluster_{label}"
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            
            clusters[cluster_id].append(all_residues[i].id)
        
        return clusters
    
    def find_cross_system_patterns(self) -> List[Dict[str, Any]]:
        """
        Find patterns that appear across different systems.
        
        Returns:
        --------
        List[Dict[str, Any]]
            Cross-system patterns
        """
        # Get all residues
        all_residues = self.repository.residues
        
        # Group by system context
        systems = {}
        for residue in all_residues:
            system = residue.system_context
            if system not in systems:
                systems[system] = []
            systems[system].append(residue)
        
        # Skip if not enough systems
        if len(systems) < 2:
            return []
        
        # Find cross-system parallels
        patterns = []
        
        # Check explicit cross-system parallels
        for residue in all_residues:
            if residue.cross_system_parallels:
                pattern = {
                    'source_residue': residue.id,
                    'source_system': residue.system_context,
                    'parallels': residue.cross_system_parallels,
                    'confidence': 'high',
                    'description': f"Explicit cross-system parallel from {residue.id}: {', '.join(residue.cross_system_parallels)}"
                }
                patterns.append(pattern)
        
        # Use text similarity to find implicit parallels
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Create system-specific documents
        system_docs = {}
        for system, residues in systems.items():
            text = ""
            for residue in residues:
                text += residue.observation + " "
                text += residue.potential_significance + " "
                if residue.relationship_to_known:
                    text += residue.relationship_to_known + " "
            system_docs[system] = text
        
        # Skip if empty documents
        if not all(system_docs.values()):
            return patterns
        
        # Calculate similarity between systems
        vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        
        try:
            tfidf_matrix = vectorizer.fit_transform(list(system_docs.values()))
            cosine_sim = cosine_similarity(tfidf_matrix)
            
            # Find similar systems
            systems_list = list(systems.keys())
            
            for i in range(len(systems_list)):
                for j in range(i + 1, len(systems_list)):
                    similarity = cosine_sim[i, j]
                    
                    if similarity > 0.3:  # Threshold for similarity
                        pattern = {
                            'source_system': systems_list[i],
                            'target_system': systems_list[j],
                            'similarity': similarity,
                            'confidence': 'medium' if similarity > 0.5 else 'low',
                            'description': f"Implicit parallel between {systems_list[i]} and {systems_list[j]} (similarity: {similarity:.2f})"
                        }
                        patterns.append(pattern)
        except Exception as e:
            logger.warning(f"Similarity calculation failed: {e}")
        
        return patterns
    
    def find_temporal_patterns(self) -> List[Dict[str, Any]]:
        """
        Find temporal patterns in residue observations.
        
        Returns:
        --------
        List[Dict[str, Any]]
            Temporal patterns
        """
        # Get all residues
        all_residues = self.repository.residues
        
        # Extract timestamps
        timestamps = []
        for residue in all_residues:
            try:
                dt = datetime.datetime.fromisoformat(residue.timestamp)
                timestamps.append((residue.id, dt))
            except:
                continue
        
        # Sort by timestamp
        timestamps.sort(key=lambda x: x[1])
        
        if len(timestamps) < 3:
            return []
        
        patterns = []
        
        # Check for acceleration/deceleration
        intervals = []
        for i in range(1, len(timestamps)):
            delta = (timestamps[i][1] - timestamps[i-1][1]).total_seconds()
            intervals.append(delta)
        
        if intervals:
            # Check trend
            trend, _, _, p_value, _ = stats.linregress(range(len(intervals)), intervals)
            
            if p_value < 0.1:  # Significant trend
                if trend > 0:
                    pattern = {
                        'type': 'deceleration',
                        'significance': p_value,
                        'description': f"Residue observations are becoming less frequent over time (p={p_value:.3f})"
                    }
                else:
                    pattern = {
                        'type': 'acceleration',
                        'significance': p_value,
                        'description': f"Residue observations are becoming more frequent over time (p={p_value:.3f})"
                    }
                patterns.append(pattern)
        
        # Check for clustering in time
        from sklearn.cluster import DBSCAN
        
        # Convert to seconds since first observation
        first_time = timestamps[0][1]
        times_seconds = [(residue_id, (dt - first_time).total_seconds()) for residue_id, dt in timestamps]
        
        # Reshape for clustering
        X = np.array([t[1] for t in times_seconds]).reshape(-1, 1)
        
        # Adaptive epsilon based on data range
        eps = (X.max() - X.min()) / 10  # 10% of range
        
        # Cluster
        clustering = DBSCAN(eps=eps, min_samples=2).fit(X)
        labels = clustering.labels_
        
        # Count clusters (excluding noise)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        
        if n_clusters > 0:
            # Group residues by cluster
            clusters = {}
            for i, label in enumerate(labels):
                if label == -1:  # Skip noise
                    continue
                
                if label not in clusters:
                    clusters[label] = []
                
                clusters[label].append(times_seconds[i][0])
            
            # Add pattern for each cluster
            for label, residue_ids in clusters.items():
                if len(residue_ids) >= 2:
                    pattern = {
                        'type': 'temporal_cluster',
                        'residue_ids': residue_ids,
                        'size': len(residue_ids),
                        'description': f"Cluster of {len(residue_ids)} residue observations in time"
                    }
                    patterns.append(pattern)
        
        return patterns
    
    def analyze_residue_evolution(self, residue_id: str) -> Dict[str, Any]:
        """
        Analyze how a residue has evolved over time.
        
        Parameters:
        -----------
        residue_id : str
            ID of the residue
        
        Returns:
        --------
        Dict[str, Any]
            Evolution analysis
        """
        residue = self.repository.get(residue_id)
        if not residue:
            return {'error': f"Residue not found: {residue_id}"}
        
        # Skip if no updates
        if not residue.update_history:
            return {'residue_id': residue_id, 'updates': 0, 'evolution': 'none'}
        
        analysis = {
            'residue_id': residue_id,
            'title': residue.title,
            'updates': len(residue.update_history),
            'first_timestamp': residue.timestamp,
            'last_update': residue.update_history[-1]['timestamp'],
            'fields_updated': set(),
            'status_evolution': []
        }
        
        # Collect updated fields
        for update in residue.update_history:
            analysis['fields_updated'].update(update['fields_updated'])
        
        # Track status evolution
        current_status = 'documented'
        analysis['status_evolution'].append({
            'timestamp': residue.timestamp,
            'status': current_status
        })
        
        for update in residue.update_history:
            if 'status' in update['fields_updated']:
                current_status = update['previous_values']['status']
                analysis['status_evolution'].append({
                    'timestamp': update['timestamp'],
                    'status': current_status
                })
        
        # Determine evolution type
        if 'status' in analysis['fields_updated']:
            if 'integrated into theory' in [s['status'] for s in analysis['status_evolution']]:
                analysis['evolution'] = 'integrated'
            elif 'active research' in [s['status'] for s in analysis['status_evolution']]:
                analysis['evolution'] = 'active'
            else:
                analysis['evolution'] = 'evolving'
        else:
            analysis['evolution'] = 'stagnant'
        
        return analysis
    
    def generate_report(self, include_clusters: bool = True,
                       include_cross_system: bool = True,
                       include_temporal: bool = True) -> Dict[str, Any]:
        """
        Generate a comprehensive report of residue analysis.
        
        Parameters:
        -----------
        include_clusters : bool, default=True
            Whether to include cluster analysis
        include_cross_system : bool, default=True
            Whether to include cross-system pattern analysis
        include_temporal : bool, default=True
            Whether to include temporal pattern analysis
        
        Returns:
        --------
        Dict[str, Any]
            Comprehensive report
        """
        report = {
            'timestamp': datetime.datetime.now().isoformat(),
            'total_residues': len(self.repository.residues),
            'system_contexts': {},
            'status_summary': {},
            'priority_summary': {},
            'analyses': {}
        }
        
        # Count by system context
        for residue in self.repository.residues:
            system = residue.system_context
            if system not in report['system_contexts']:
                report['system_contexts'][system] = 0
            report['system_contexts'][system] += 1
        
        # Count by status
        for residue in self.repository.residues:
            status = residue.status
            if status not in report['status_summary']:
                report['status_summary'][status] = 0
            report['status_summary'][status] += 1
        
        # Count by priority
        for residue in self.repository.residues:
            priority = residue.priority
            if priority not in report['priority_summary']:
                report['priority_summary'][priority] = 0
            report['priority_summary'][priority] += 1
        
        # Include requested analyses
        if include_clusters:
            report['analyses']['clusters'] = self.find_clusters()
        
        if include_cross_system:
            report['analyses']['cross_system_patterns'] = self.find_cross_system_patterns()
        
        if include_temporal:
            report['analyses']['temporal_patterns'] = self.find_temporal_patterns()
        
        # Add recommendations
        report['recommendations'] = self._generate_recommendations(report)
        
        return report
    
    def _generate_recommendations(self, report: Dict[str, Any]) -> List[str]:
        """
        Generate recommendations based on analysis.
        
        Parameters:
        -----------
        report : Dict[str, Any]
            Analysis report
        
        Returns:
        --------
        List[str]
            Recommendations
        """
        recommendations = []
        
        # Check for unexamined residues
        if 'documented' in report['status_summary'] and report['status_summary']['documented'] > 0:
            documented_count = report['status_summary']['documented']
            recommendations.append(f"Examine {documented_count} residues currently in 'documented' status")
        
        # Check for high priority residues
        if 'high' in report['priority_summary'] and report['priority_summary']['high'] > 0:
            high_priority_count = report['priority_summary']['high']
            recommendations.append(f"Focus on {high_priority_count} high-priority residues")
        
        # Check for cross-system patterns
        if 'cross_system_patterns' in report['analyses'] and report['analyses']['cross_system_patterns']:
            pattern_count = len(report['analyses']['cross_system_patterns'])
            recommendations.append(f"Investigate {pattern_count} cross-system patterns for potential universal principles")
        
        # Check for temporal clusters
        if 'temporal_patterns' in report['analyses']:
            temporal_patterns = report['analyses']['temporal_patterns']
            cluster_patterns = [p for p in temporal_patterns if p['type'] == 'temporal_cluster']
            if cluster_patterns:
                cluster_count = len(cluster_patterns)
                recommendations.append(f"Examine {cluster_count} temporal clusters for potential triggering events")
        
        return recommendations
    
    def visualize_residue_network(self, figsize: Tuple[int, int] = (12, 10),
                                 save_path: Optional[str] = None) -> plt.Figure:
        """
        Visualize network of residues and their relationships.
        
        Parameters:
        -----------
        figsize : Tuple[int, int], default=(12, 10)
            Figure size
        save_path : str, optional
            Path to save the visualization
        
        Returns:
        --------
        plt.Figure
            Figure containing the visualization
        """
        try:
            import networkx as nx
            
            # Create graph
            G = nx.Graph()
            
            # Add nodes for residues
            for residue in self.repository.residues:
                G.add_node(residue.id, 
                          title=residue.title, 
                          system=residue.system_context,
                          status=residue.status,
                          priority=residue.priority)
            
            # Add edges for explicit relationships
            for residue in self.repository.residues:
                # Add edges based on cross-system parallels
                for parallel in residue.cross_system_parallels:
                    # Try to find matching residue
                    for other in self.repository.residues:
                        if other.id != residue.id and (
                            parallel in other.title or 
                            any(parallel in h for h in other.initial_hypotheses) or
                            parallel in other.observation
                        ):
                            G.add_edge(residue.id, other.id, type='parallel')
            
            # Add edges based on text similarity
            texts = {}
            for residue in self.repository.residues:
                text = residue.observation + " " + residue.potential_significance
                texts[residue.id] = text
            
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            
            vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(list(texts.values()))
            cosine_sim = cosine_similarity(tfidf_matrix)
            
            residue_ids = list(texts.keys())
            for i in range(len(residue_ids)):
                for j in range(i + 1, len(residue_ids)):
                    similarity = cosine_sim[i, j]
                    if similarity > 0.3:  # Threshold
                        G.add_edge(residue_ids[i], residue_ids[j], 
                                  weight=similarity, type='similarity')
            
            # Create figure
            fig, ax = plt.subplots(figsize=figsize)
            
            # Set node colors by system
            systems = set(nx.get_node_attributes(G, 'system').values())
            system_colors = plt.cm.tab10(np.linspace(0, 1, len(systems)))
            system_color_map = dict(zip(systems, system_colors))
            
            node_colors = [system_color_map[G.nodes[n]['system']] for n in G.nodes]
            
            # Set node sizes by priority
            priority_sizes = {'low': 100, 'medium': 300, 'high': 500}
            node_sizes = [priority_sizes.get(G.nodes[n]['priority'], 200) for n in G.nodes]
            
            # Layout
            pos = nx.spring_layout(G, seed=42)
            
            # Draw nodes
            nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                                  node_size=node_sizes, alpha=0.8, ax=ax)
            
            # Draw edges
            parallel_edges = [(u, v) for u, v, d in G.edges(data=True) if d['type'] == 'parallel']
            similarity_edges = [(u, v) for u, v, d in G.edges(data=True) if d['type'] == 'similarity']
            
            nx.draw_networkx_edges(G, pos, edgelist=parallel_edges, 
                                  width=2, edge_color='red', ax=ax)
            nx.draw_networkx_edges(G, pos, edgelist=similarity_edges, 
                                  width=1, edge_color='blue', alpha=0.5, ax=ax)
            
            # Draw labels for important nodes
            # Only label nodes with high priority or high degree
            important_nodes = [n for n in G.nodes 
                              if G.nodes[n]['priority'] == 'high' or G.degree(n) > 2]
            
            label_dict = {n: G.nodes[n]['title'] for n in important_nodes}
            nx.draw_networkx_labels(G, pos, labels=label_dict, font_size=8, 
                                   font_weight='bold', ax=ax)
            
            # Add legend for systems
            handles = []
            for system, color in system_color_map.items():
                patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, 
                                  markersize=10, label=system)
                handles.append(patch)
            
            # Add legend for priority sizes
            for priority, size in priority_sizes.items():
                patch = plt.Line2D([0], [0], marker='o', color='gray', 
                                  markersize=np.sqrt(size/30), 
                                  label=f"{priority.capitalize()} Priority")
                handles.append(patch)
            
            # Add legend for edge types
            handles.append(plt.Line2D([0], [0], color='red', lw=2, label='Explicit Parallel'))
            handles.append(plt.Line2D([0], [0], color='blue', lw=1, alpha=0.5, 
                                     label='Similarity Link'))
            
            ax.legend(handles=handles, loc='upper right', title='Legend')
            
            # Set title and remove axis
            ax.set_title('Symbolic Residue Network', fontsize=16)
            ax.axis('off')
            
            # Adjust layout
            plt.tight_layout()
            
            # Save if requested
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            return fig
        
        except ImportError:
            logger.warning("NetworkX or sklearn not available for visualization")
            
            # Create empty figure with message
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, "NetworkX or sklearn not available for visualization", 
                   ha='center', va='center', fontsize=14)
            ax.axis('off')
            
            return fig
    
    def visualize_temporal_evolution(self, figsize: Tuple[int, int] = (12, 8),
                                    save_path: Optional[str] = None) -> plt.Figure:
        """
        Visualize temporal evolution of residue observations.
        
        Parameters:
        -----------
        figsize : Tuple[int, int], default=(12, 8)
            Figure size
        save_path : str, optional
            Path to save the visualization
        
        Returns:
        --------
        plt.Figure
            Figure containing the visualization
        """
        # Extract timestamps and systems
        data = []
        for residue in self.repository.residues:
            try:
                dt = datetime.datetime.fromisoformat(residue.timestamp)
                data.append({
                    'id': residue.id,
                    'timestamp': dt,
                    'system': residue.system_context,
                    'priority': residue.priority,
                    'status': residue.status,
                    'title': residue.title
                })
            except:
                continue
        
        # Sort by timestamp
        data.sort(key=lambda x: x['timestamp'])
        
        if not data:
            # Create empty figure with message
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, "No timestamped residues available", 
                   ha='center', va='center', fontsize=14)
            ax.axis('off')
            
            return fig
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, 
                                      gridspec_kw={'height_ratios': [3, 1]})
        
        # Get unique systems and priorities
        systems = sorted(set(d['system'] for d in data))
        priorities = ['high', 'medium', 'low']
        
        # Create system and priority color maps
        system_colors = plt.cm.tab10(np.linspace(0, 1, len(systems)))
        system_color_map = dict(zip(systems, system_colors))
        
        priority_colors = {'high': 'red', 'medium': 'blue', 'low': 'green'}
        
        # Create timeline
        for i, d in enumerate(data):
            # Add system point
            ax1.scatter(d['timestamp'], i, 
                       c=[system_color_map[d['system']]], 
                       s=100, alpha=0.7, 
                       edgecolors='black', linewidths=1)
            
            # Add priority marker
            marker = 'o' if d['priority'] == 'medium' else ('*' if d['priority'] == 'high' else 's')
            ax1.scatter(d['timestamp'], i, 
                       c=priority_colors[d['priority']], 
                       s=150, alpha=0.5, marker=marker)
        
        # Add labels
        ax1.set_yticks(range(len(data)))
        ax1.set_yticklabels([f"{d['id']}: {d['title'][:20]}..." for d in data])
        
        ax1.set_title('Symbolic Residue Timeline', fontsize=16)
        ax1.grid(True, alpha=0.3)
        
        # Create histogram of observations over time
        timestamps = [d['timestamp'] for d in data]
        min_time = min(timestamps)
        max_time = max(timestamps)
        
        # Determine bin width based on time range
        time_range = (max_time - min_time).total_seconds()
        if time_range < 86400:  # Less than a day
            bin_width = datetime.timedelta(hours=1)
            date_format = '%H:%M'
        elif time_range < 604800:  # Less than a week
            bin_width = datetime.timedelta(days=1)
            date_format = '%m-%d'
        elif time_range < 2592000:  # Less than a month
            bin_width = datetime.timedelta(weeks=1)
            date_format = '%m-%d'
        else:
            bin_width = datetime.timedelta(days=30)
            date_format = '%Y-%m'
        
        # Create bins
        bin_count = int(time_range / bin_width.total_seconds()) + 1
        bins = [min_time + bin_width * i for i in range(bin_count + 1)]
        
        # Count observations per bin
        counts = np.zeros(bin_count)
        for dt in timestamps:
            bin_idx = int((dt - min_time).total_seconds() / bin_width.total_seconds())
            if 0 <= bin_idx < bin_count:
                counts[bin_idx] += 1
        
        # Plot histogram
        bin_centers = [min_time + bin_width * (i + 0.5) for i in range(bin_count)]
        ax2.bar(bin_centers, counts, width=bin_width.total_seconds() * 0.8, 
               alpha=0.7, color='skyblue', edgecolor='black')
        
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Count')
        ax2.set_title('Observation Frequency')
        ax2.grid(True, alpha=0.3)
        
        # Format dates
        import matplotlib.dates as mdates
        date_format = mdates.DateFormatter(date_format)
        ax1.xaxis.set_major_formatter(date_format)
        ax2.xaxis.set_major_formatter(date_format)
        
        plt.xticks(rotation=45)
        
        # Add legends
        # System legend
        system_handles = []
        for system, color in system_color_map.items():
            patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, 
                              markersize=10, label=system)
            system_handles.append(patch)
        
        # Priority legend
        priority_handles = []
        for priority in priorities:
            marker = 'o' if priority == 'medium' else ('*' if priority == 'high' else 's')
            patch = plt.Line2D([0], [0], marker=marker, color=priority_colors[priority], 
                              markersize=10, label=f"{priority.capitalize()} Priority")
            priority_handles.append(patch)
        
        # Add legends
        ax1.legend(handles=system_handles, loc='upper left', 
                  title='Systems', bbox_to_anchor=(1.01, 1))
        ax1.legend(handles=priority_handles, loc='upper left', 
                  title='Priorities', bbox_to_anchor=(1.01, 0.5))
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def visualize_status_transitions(self, residue_id: str = None,
                                   figsize: Tuple[int, int] = (10, 6),
                                   save_path: Optional[str] = None) -> plt.Figure:
        """
        Visualize status transitions of residues.
        
        Parameters:
        -----------
        residue_id : str, optional
            ID of specific residue to visualize, or None for all
        figsize : Tuple[int, int], default=(10, 6)
            Figure size
        save_path : str, optional
            Path to save the visualization
        
        Returns:
        --------
        plt.Figure
            Figure containing the visualization
        """
        # Get residues to analyze
        if residue_id:
            residues = [self.repository.get(residue_id)]
            if not residues[0]:
                # Create empty figure with message
                fig, ax = plt.subplots(figsize=figsize)
                ax.text(0.5, 0.5, f"Residue not found: {residue_id}", 
                       ha='center', va='center', fontsize=14)
                ax.axis('off')
                
                return fig
        else:
            residues = self.repository.residues
        
        # Analyze residue evolution
        evolutions = []
        for residue in residues:
            evol = self.analyze_residue_evolution(residue.id)
            if 'error' not in evol:
                evolutions.append(evol)
        
        if not evolutions:
            # Create empty figure with message
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, "No residue evolution data available", 
                   ha='center', va='center', fontsize=14)
            ax.axis('off')
            
            return fig
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Set up Sankey diagram
        try:
            from matplotlib.sankey import Sankey
            
            # Count status transitions
            transitions = {}
            statuses = set()
            
            for evol in evolutions:
                if 'status_evolution' in evol and len(evol['status_evolution']) >= 2:
                    # Get status sequence
                    sequence = [s['status'] for s in evol['status_evolution']]
                    statuses.update(sequence)
                    
                    # Count transitions
                    for i in range(len(sequence) - 1):
                        source = sequence[i]
                        target = sequence[i + 1]
                        
                        if source != target:
                            key = (source, target)
                            if key not in transitions:
                                transitions[key] = 0
                            transitions[key] += 1
            
            # Skip if no transitions
            if not transitions:
                ax.text(0.5, 0.5, "No status transitions found", 
                       ha='center', va='center', fontsize=14)
                ax.axis('off')
                
                return fig
            
            # Create status index mapping
            statuses = sorted(statuses)
            status_indices = {status: i for i, status in enumerate(statuses)}
            
            # Prepare Sankey data
            flows = []
            labels = []
            for (source, target), count in transitions.items():
                flows.append(count)
                labels.append(f"{source} → {target}")
            
            # Create Sankey diagram
            sankey = Sankey(ax=ax, scale=0.01, offset=0.2, head_angle=120, 
                           format='%.0f', unit='', gap=0.5)
            
            # Add flows
            for i, ((source, target), count) in enumerate(transitions.items()):
                sankey.add(flows=[count], 
                          orientations=[0], 
                          labels=[f"{source} → {target}"],
                          pathlengths=[0.5])
            
            # Finish diagram
            sankey.finish()
            
            ax.set_title('Residue Status Transitions', fontsize=16)
            ax.axis('off')
            
        except ImportError:
            # Alternative visualization if Sankey not available
            logger.warning("Matplotlib Sankey not available, using alternative visualization")
            
            # Get all statuses and count residues in each
            status_counts = {}
            for residue in residues:
                status = residue.status
                if status not in status_counts:
                    status_counts[status] = 0
                status_counts[status] += 1
            
            # Create bar chart
            statuses = list(status_counts.keys())
            counts = [status_counts[s] for s in statuses]
            
            ax.bar(statuses, counts, color='skyblue', edgecolor='black')
            
            ax.set_title('Residue Status Distribution', fontsize=16)
            ax.set_xlabel('Status')
            ax.set_ylabel('Count')
            
            # Rotate x labels if needed
            if len(statuses) > 3:
                plt.xticks(rotation=45, ha='right')
            
            ax.grid(True, alpha=0.3)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig

def create_residue_from_template(template_path: str = None, **kwargs) -> SymbolicResidue:
    """
    Create a symbolic residue from template.
    
    Parameters:
    -----------
    template_path : str, optional
        Path to template JSON file
    **kwargs
        Fields to override template values
    
    Returns:
    --------
    SymbolicResidue
        Created residue
    """
    # Default template
    template = {
        "observer": "researcher",
        "system_context": "",
        "title": "",
        "observation": "",
        "context": {},
        "repeatability": "unknown",
        "documentation": [],
        "initial_hypotheses": [],
        "relationship_to_known": "",
        "cross_system_parallels": [],
        "potential_significance": "",
        "status": "documented",
        "investigation_steps": [],
        "open_questions": [],
        "next_steps": [],
        "priority": "medium",
        "recursive_nature": "",
        "meta_implications": "",
        "researcher_effect": "",
        "personal_reflection": "",
        "tags": []
    }
    
    # Load template if provided
    if template_path:
        try:
            with open(template_path, 'r') as f:
                loaded_template = json.load(f)
                template.update(loaded_template)
        except Exception as e:
            logger.warning(f"Failed to load template: {e}")
    
    # Override with kwargs
    template.update(kwargs)
    
    # Create residue
    return SymbolicResidue(**template)

def detect_anomalies(data: np.ndarray, threshold_factor: float = 3.0) -> List[int]:
    """
    Detect anomalies in data.
    
    Parameters:
    -----------
    data : np.ndarray
        Data to analyze
    threshold_factor : float, default=3.0
        Factor for threshold calculation
    
    Returns:
    --------
    List[int]
        Indices of anomalies
    """
    if len(data) < 2:
        return []
    
    # Calculate z-scores
    mean = np.mean(data)
    std = np.std(data)
    
    if std == 0:
        return []
    
    z_scores = np.abs((data - mean) / std)
    
    # Detect anomalies
    anomalies = np.where(z_scores > threshold_factor)[0]
    
    return list(anomalies)

def detect_phase_transitions(data: np.ndarray, window_size: int = 5) -> List[int]:
    """
    Detect phase transitions in data.
    
    Parameters:
    -----------
    data : np.ndarray
        Data to analyze
    window_size : int, default=5
        Size of sliding window
    
    Returns:
    --------
    List[int]
        Indices of phase transitions
    """
    if len(data) < window_size * 2:
        return []
    
    # Calculate moving average
    window = np.ones(window_size) / window_size
    moving_avg = np.convolve(data, window, mode='valid')
    
    # Calculate derivatives
    derivatives = np.diff(moving_avg)
    
    # Calculate standard deviation of derivatives
    std_dev = np.std(derivatives)
    
    if std_dev == 0:
        return []
    
    # Find significant changes
    transitions = []
    for i in range(1, len(derivatives)):
        if abs(derivatives[i] - derivatives[i-1]) > 2 * std_dev:
            # Adjust index to original data
            transitions.append(i + window_size)
    
    return transitions

def analyze_system(data: Dict[str, np.ndarray], timestamps: List[datetime.datetime] = None) -> Dict[str, Any]:
    """
    Analyze system data for patterns and anomalies.
    
    Parameters:
    -----------
    data : Dict[str, np.ndarray]
        System data (metric -> values)
    timestamps : List[datetime.datetime], optional
        Timestamps for data points
    
    Returns:
    --------
    Dict[str, Any]
        Analysis results
    """
    results = {
        'anomalies': {},
        'transitions': {},
        'correlations': {},
        'cycles': {}
    }
    
    # Analyze each metric
    for metric, values in data.items():
        # Skip if not enough data
        if len(values) < 5:
            continue
        
        # Detect anomalies
        anomalies = detect_anomalies(values)
        if anomalies:
            results['anomalies'][metric] = anomalies
        
        # Detect phase transitions
        transitions = detect_phase_transitions(values)
        if transitions:
            results['transitions'][metric] = transitions
        
        # Detect cycles
        if len(values) >= 10:
            from scipy import signal
            
            # Normalize values
            normalized = values - np.mean(values)
            if np.std(normalized) > 0:
                normalized = normalized / np.std(normalized)
                
                # Calculate autocorrelation
                autocorr = signal.correlate(normalized, normalized, mode='full')
                autocorr = autocorr[len(autocorr)//2:]
                
                # Find peaks
                peaks, _ = signal.find_peaks(autocorr, height=0.3)
                
                if len(peaks) > 1:
                    # Calculate cycle period
                    if peaks[0] == 0:
                        peaks = peaks[1:]
                    
                    period = np.mean(np.diff(peaks))
                    
                    if period > 1:
                        results['cycles'][metric] = {
                            'period': period,
                            'strength': autocorr[int(period)] / autocorr[0]
                        }
    
    # Calculate correlations between metrics
    metrics = list(data.keys())
    if len(metrics) >= 2:
        for i in range(len(metrics)):
            for j in range(i + 1, len(metrics)):
                metric1 = metrics[i]
                metric2 = metrics[j]
                
                values1 = data[metric1]
                values2 = data[metric2]
                
                # Ensure same length
                min_len = min(len(values1), len(values2))
                if min_len >= 5:
                    values1 = values1[:min_len]
                    values2 = values2[:min_len]
                    
                    # Calculate correlation
                    corr, p_value = stats.pearsonr(values1, values2)
                    
                    if abs(corr) > 0.5 and p_value < 0.05:
                        results['correlations'][(metric1, metric2)] = {
                            'correlation': corr,
                            'p_value': p_value
                        }
    
    # Analyze temporal patterns if timestamps provided
    if timestamps and len(timestamps) == len(next(iter(data.values()))):
        results['temporal'] = _analyze_temporal_patterns(data, timestamps)
    
    return results

def _analyze_temporal_patterns(data: Dict[str, np.ndarray], timestamps: List[datetime.datetime]) -> Dict[str, Any]:
    """
    Analyze temporal patterns in data.
    
    Parameters:
    -----------
    data : Dict[str, np.ndarray]
        System data (metric -> values)
    timestamps : List[datetime.datetime]
        Timestamps for data points
    
    Returns:
    --------
    Dict[str, Any]
        Temporal analysis results
    """
    results = {
        'trends': {},
        'seasonality': {},
        'temporal_transitions': {}
    }
    
    # Convert timestamps to seconds since start
    start_time = min(timestamps)
    time_seconds = [(dt - start_time).total_seconds() for dt in timestamps]
    
    # Analyze each metric
    for metric, values in data.items():
        # Skip if not enough data
        if len(values) < 10:
            continue
        
        # Detect trend
        trend, intercept, r_value, p_value, std_err = stats.linregress(time_seconds, values)
        
        if p_value < 0.05:
            results['trends'][metric] = {
                'slope': trend,
                'intercept': intercept,
                'r_squared': r_value**2,
                'p_value': p_value
            }
        
        # Detect seasonal patterns
        # This is a simplified approach; for more complex seasonality detection,
        # consider using seasonal decomposition (e.g., from statsmodels)
        try:
            # Check for daily pattern
            day_seconds = 86400
            if max(time_seconds) >= day_seconds:
                # Convert to time of day
                day_time = [t % day_seconds for t in time_seconds]
                
                # Group by time of day (hourly bins)
                bins = 24
                hourly_values = [[] for _ in range(bins)]
                
                for i, t in enumerate(day_time):
                    bin_idx = int(t / (day_seconds / bins))
                    if bin_idx < bins:
                        hourly_values[bin_idx].append(values[i])
                
                # Calculate mean for each hour
                hourly_means = [np.mean(v) if v else 0 for v in hourly_values]
                
                # Check variation in hourly means
                if np.std(hourly_means) > 0.1 * np.mean(hourly_means):
                    # Significant hourly variation
                    results['seasonality'][metric] = {
                        'pattern': 'daily',
                        'variation': np.std(hourly_means) / np.mean(hourly_means)
                    }
        except:
            pass
        
        # Detect temporal transitions
        transitions = detect_phase_transitions(values)
        if transitions:
            results['temporal_transitions'][metric] = [
                {
                    'index': idx,
                    'timestamp': timestamps[idx] if idx < len(timestamps) else None
                }
                for idx in transitions if idx < len(values)
            ]
    
    return results

# Example usage if run directly
if __name__ == "__main__":
    # Create repository
    repo = ResidueRepository("./residue")
    
    # Create sample residue
    residue = create_residue_from_template(
        observer="David Kimai",
        system_context="Langton's Ant",
        title="Transient Spiral Echoes",
        observation="Brief appearances of spiral-like structures before dissolving",
        potential_significance="May indicate temporary attractor states",
        priority="high"
    )
    
    # Add to repository
    repo.add(residue)
    
    # Create analyzer
    analyzer = ResidueAnalyzer(repo)
    
    # Generate report
    report = analyzer.generate_report()
    
    # Print report
    print(json.dumps(report, indent=2, default=str))
