import time
import pm4py
from neo4j import GraphDatabase
import os

#SETUP
experiment_name = 'order-management'
file_path = os.path.join("ocel2", experiment_name + ".jsonocel")


lbl_event = 'Event'
lbl_entity =  'Entity'
lbl_rel = 'REL'
lbl_corr='CORR'
lbl_df = 'DF'
lbl_derived = 'DERIVED'

lbl_meta_node_log = 'node:Log'  
lbl_meta_node_class = 'node:Class' 
lbl_meta_node_event = 'node:Event' 
lbl_meta_node_entity = 'node:Entity'
lbl_meta_node_snapshot = 'node:Snapshot' 

lbl_meta_node_entity_reified = 'node:Reified_Entity' 
lbl_meta_node_snapshot_reified = 'node:Reified_Snapshot'

lbl_meta_rel_log_has_event = 'rel:has' 
lbl_meta_rel_event_observed_class = 'rel:observed'
lbl_meta_rel_entity_snapshot_snapshot = 'rel:snapshot'  
lbl_meta_rel_snapshot_rel_update_snapshot = 'rel:rel:SnapshotUpdate'  
lbl_meta_rel_entity_rel_entity = 'rel:rel:Entity'
lbl_meta_rel_snapshot_rel_snapshot = 'rel:rel:Snapshot' 

lbl_meta_rel_derived = 'rel:derived' 


lbl_meta_rel_event_corr = 'rel:corr' 
lbl_meta_rel_event_corr_entity = 'rel:corr:Entity'
lbl_meta_rel_event_corr_entity_reified = 'rel:corr:ReifiedEntity' 

lbl_meta_rel_event_corr_snapshot  = 'rel:corr:Snapshot'
lbl_meta_rel_event_corr_snapshot_reified  ='rel:corr:ReifiedSnapshot' 

lbl_meta_rel_event_df_entity_event  ='rel_Event-df[entity]->Event'
lbl_meta_rel_event_df_snapshot_event  ='rel_Event-df[snapshot]->Event'
lbl_meta_rel_event_df_event='rel:df' 

meta_time = {}
URI  = 'bolt://localhost:7687'
AUTH = ('neo4j', '12341234')

with GraphDatabase.driver(URI, auth=AUTH) as driver:
    driver.verify_connectivity()
    
ocel = pm4py.read.read_ocel2_json(file_path)

with GraphDatabase.driver(URI, auth=AUTH) as driver:
    driver.execute_query('MATCH (a) DETACH DELETE a')
    
#Event Nodes
action = lbl_meta_node_event
start = time.time()


with GraphDatabase.driver(URI, auth=AUTH) as driver:
    for idx, row in ocel.events.iterrows():
        driver.execute_query("CREATE (:"+lbl_event+" {EventID: '"+
            row[ocel.event_id_column]+
            "', timestamp: datetime('"+
            str(row[ocel.event_timestamp].strftime('%Y-%m-%dT%H:%M')+':00.000+0100')+
            "'), Activity:'"+
            row[ocel.event_activity]+
            "'})")

end = time.time()
print(end - start)
meta_time[action] =  end - start

#Entity Nodes
action = lbl_meta_node_entity
start = time.time()

cols = list(ocel.objects.columns)
def map(n):
    return '`'+n.replace(ocel.object_id_column, "ID").replace(ocel.object_type_column, "EntityType")+'`'

with GraphDatabase.driver(URI, auth=AUTH) as driver:
    for idx, rows in ocel.objects.fillna('').iterrows():
            atts = [map(c)+":'"+str(rows[c])+"'" for c in cols]
            res = ""
            for a in atts:
                res = res + a + ", "
        
                
            driver.execute_query("CREATE (:"+lbl_entity+" {"+
                  res[:-2] +
                 "})"
                 )
        
end = time.time()
print(end - start)
meta_time[action] =  end - start

#REL Edges
action = lbl_meta_rel_entity_rel_entity
start = time.time()

with GraphDatabase.driver(URI, auth=AUTH) as driver:
    for idx,row in ocel.o2o.iterrows():
        o1 = row[ocel.object_id_column]
        o2 = row[ocel.object_id_column+'_2']
        q  = row[ocel.qualifier]
    
        driver.execute_query(
            "MATCH (o1:"+lbl_entity+" {ID: '"+str(o1)+"'}), (o2:"+lbl_entity+" {ID: '"+str(o2)+"'}) MERGE (o1)-[:"+lbl_rel +" {qual:'"+str(q)+"'}]->(o2)" 
        )

end = time.time()
print(end - start)
meta_time[action] =  end - start

#CORR Edges
action = lbl_meta_rel_event_corr_entity
start = time.time()

with GraphDatabase.driver(URI, auth=AUTH) as driver:
    for idx,row in ocel.relations.iterrows():
        e = row[ocel.event_id_column]
        o = row[ocel.object_id_column]
        q  = row[ocel.qualifier]

        driver.execute_query(
            "MATCH (e:"+lbl_event+" {EventID:'" + str(e) + "'}), (o:"+lbl_entity+" {ID:'"+str(o)+"'}) " +
            "MERGE (e)-[:"+lbl_corr+"]->(o)"
        )

end = time.time()
print(end - start)
meta_time[action] =  end - start

#DF Edges
action = lbl_meta_rel_event_df_event
start = time.time()

with GraphDatabase.driver(URI, auth=AUTH) as driver:
    
    driver.execute_query(
    "MATCH (e:"+lbl_event+")-[:"+lbl_corr+"]->(n)  " +
    "WITH n, e order by e.timestamp  " +
    "WITH n, collect(e) as es, range(0, size(collect(e))-2) as esn " +
    "UNWIND esn as i  " +
    "MATCH (a), (b)  " +
    "WHERE a=es[i] and b=es[i+1]  " +
    "MERGE (a)-[:"+lbl_df+" {EntityType:n.EntityType, EntityID:n.ID}]->(b) "
    )

# removing parallel dfs

with GraphDatabase.driver(URI, auth=AUTH) as driver:
    driver.execute_query(
        "MATCH ()-[r:"+lbl_df+"]->() " +
        "SET r.addNewKnowledge = TRUE " 
    )
    driver.execute_query(
        "MATCH (n)<-[:"+lbl_corr+"]-(e1:"+lbl_event+")-[r1:"+lbl_df+" {EntityID:n.ID}]->(e2:"+lbl_event+") " +
        "MATCH (n)<-[:"+lbl_derived+"]-(rn)<--(e1)-[r2:"+lbl_df+"  {EntityID:rn.ID}]->(e2) " +
        "SET r2.addNewKnowledge = FALSE " 
    )
    driver.execute_query(
        "MATCH ()-[rb:"+lbl_df+" {addNewKnowledge:TRUE}]->()-[r:"+lbl_df+"  {EntityID:rb.ID}]->()-[ra:"+lbl_df+"  {EntityID:r.ID, addNewKnowledge:TRUE}]->() " +
        "SET r.addNewKnowledge=TRUE " 
    )
    driver.execute_query(
        "MATCH ()-[r:"+lbl_df+" {addNewKnowledge:FALSE}]->() " +
        "DELETE r " 
    )
    driver.execute_query(
        "MATCH ()-[r:"+lbl_df+"]->() " +
        "REMOVE  r.addNewKnowledge " 
    )


end = time.time()
print(end - start)
meta_time[action] =  end - start

