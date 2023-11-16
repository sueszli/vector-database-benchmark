from django.contrib.contenttypes.models import ContentType
from django.test import TestCase
from netaddr import IPNetwork
from dcim.choices import InterfaceTypeChoices
from dcim.models import Device, DeviceRole, DeviceType, Interface, Location, Manufacturer, Rack, Region, Site, SiteGroup
from ipam.choices import *
from ipam.filtersets import *
from ipam.models import *
from utilities.testing import ChangeLoggedFilterSetTests, create_test_device, create_test_virtualmachine
from virtualization.models import Cluster, ClusterGroup, ClusterType, VirtualMachine, VMInterface
from tenancy.models import Tenant, TenantGroup

class ASNRangeTestCase(TestCase, ChangeLoggedFilterSetTests):
    queryset = ASNRange.objects.all()
    filterset = ASNRangeFilterSet

    @classmethod
    def setUpTestData(cls):
        if False:
            i = 10
            return i + 15
        rirs = [RIR(name='RIR 1', slug='rir-1'), RIR(name='RIR 2', slug='rir-2'), RIR(name='RIR 3', slug='rir-3')]
        RIR.objects.bulk_create(rirs)
        tenants = [Tenant(name='Tenant 1', slug='tenant-1'), Tenant(name='Tenant 2', slug='tenant-2')]
        Tenant.objects.bulk_create(tenants)
        asn_ranges = (ASNRange(name='ASN Range 1', slug='asn-range-1', rir=rirs[0], tenant=None, start=65000, end=65009, description='aaa'), ASNRange(name='ASN Range 2', slug='asn-range-2', rir=rirs[1], tenant=tenants[0], start=65010, end=65019, description='bbb'), ASNRange(name='ASN Range 3', slug='asn-range-3', rir=rirs[2], tenant=tenants[1], start=65020, end=65029, description='ccc'))
        ASNRange.objects.bulk_create(asn_ranges)

    def test_name(self):
        if False:
            return 10
        params = {'name': ['ASN Range 1', 'ASN Range 2']}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 2)

    def test_rir(self):
        if False:
            return 10
        rirs = RIR.objects.all()[:2]
        params = {'rir_id': [rirs[0].pk, rirs[1].pk]}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 2)
        params = {'rir': [rirs[0].slug, rirs[1].slug]}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 2)

    def test_tenant(self):
        if False:
            return 10
        tenants = Tenant.objects.all()[:2]
        params = {'tenant_id': [tenants[0].pk, tenants[1].pk]}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 2)
        params = {'tenant': [tenants[0].slug, tenants[1].slug]}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 2)

    def test_start(self):
        if False:
            print('Hello World!')
        params = {'start': [65000, 65010]}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 2)

    def test_end(self):
        if False:
            return 10
        params = {'end': [65009, 65019]}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 2)

    def test_description(self):
        if False:
            print('Hello World!')
        params = {'description': ['aaa', 'bbb']}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 2)

class ASNTestCase(TestCase, ChangeLoggedFilterSetTests):
    queryset = ASN.objects.all()
    filterset = ASNFilterSet

    @classmethod
    def setUpTestData(cls):
        if False:
            while True:
                i = 10
        rirs = [RIR(name='RIR 1', slug='rir-1', is_private=True), RIR(name='RIR 2', slug='rir-2', is_private=True), RIR(name='RIR 3', slug='rir-3', is_private=True)]
        RIR.objects.bulk_create(rirs)
        sites = [Site(name='Site 1', slug='site-1'), Site(name='Site 2', slug='site-2'), Site(name='Site 3', slug='site-3')]
        Site.objects.bulk_create(sites)
        tenants = [Tenant(name='Tenant 1', slug='tenant-1'), Tenant(name='Tenant 2', slug='tenant-2'), Tenant(name='Tenant 3', slug='tenant-3'), Tenant(name='Tenant 4', slug='tenant-4'), Tenant(name='Tenant 5', slug='tenant-5')]
        Tenant.objects.bulk_create(tenants)
        asns = (ASN(asn=65001, rir=rirs[0], tenant=tenants[0], description='aaa'), ASN(asn=65002, rir=rirs[1], tenant=tenants[1], description='bbb'), ASN(asn=65003, rir=rirs[2], tenant=tenants[2], description='ccc'), ASN(asn=4200000000, rir=rirs[0], tenant=tenants[0]), ASN(asn=4200000001, rir=rirs[1], tenant=tenants[1]), ASN(asn=4200000002, rir=rirs[2], tenant=tenants[2]))
        ASN.objects.bulk_create(asns)
        asns[0].sites.set([sites[0]])
        asns[1].sites.set([sites[1]])
        asns[2].sites.set([sites[2]])
        asns[3].sites.set([sites[0]])
        asns[4].sites.set([sites[1]])
        asns[5].sites.set([sites[2]])

    def test_asn(self):
        if False:
            while True:
                i = 10
        params = {'asn': [65001, 4200000000]}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 2)

    def test_tenant(self):
        if False:
            print('Hello World!')
        tenants = Tenant.objects.all()[:2]
        params = {'tenant_id': [tenants[0].pk, tenants[1].pk]}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 4)
        params = {'tenant': [tenants[0].slug, tenants[1].slug]}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 4)

    def test_rir(self):
        if False:
            return 10
        rirs = RIR.objects.all()[:2]
        params = {'rir_id': [rirs[0].pk, rirs[1].pk]}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 4)
        params = {'rir': [rirs[0].slug, rirs[1].slug]}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 4)

    def test_site(self):
        if False:
            return 10
        sites = Site.objects.all()[:2]
        params = {'site_id': [sites[0].pk, sites[1].pk]}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 4)
        params = {'site': [sites[0].slug, sites[1].slug]}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 4)

    def test_description(self):
        if False:
            i = 10
            return i + 15
        params = {'description': ['aaa', 'bbb']}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 2)

class VRFTestCase(TestCase, ChangeLoggedFilterSetTests):
    queryset = VRF.objects.all()
    filterset = VRFFilterSet

    @classmethod
    def setUpTestData(cls):
        if False:
            for i in range(10):
                print('nop')
        route_targets = (RouteTarget(name='65000:1001'), RouteTarget(name='65000:1002'), RouteTarget(name='65000:1003'))
        RouteTarget.objects.bulk_create(route_targets)
        tenant_groups = (TenantGroup(name='Tenant group 1', slug='tenant-group-1'), TenantGroup(name='Tenant group 2', slug='tenant-group-2'), TenantGroup(name='Tenant group 3', slug='tenant-group-3'))
        for tenantgroup in tenant_groups:
            tenantgroup.save()
        tenants = (Tenant(name='Tenant 1', slug='tenant-1', group=tenant_groups[0]), Tenant(name='Tenant 2', slug='tenant-2', group=tenant_groups[1]), Tenant(name='Tenant 3', slug='tenant-3', group=tenant_groups[2]))
        Tenant.objects.bulk_create(tenants)
        vrfs = (VRF(name='VRF 1', rd='65000:100', tenant=tenants[0], enforce_unique=False, description='foobar1'), VRF(name='VRF 2', rd='65000:200', tenant=tenants[0], enforce_unique=False, description='foobar2'), VRF(name='VRF 3', rd='65000:300', tenant=tenants[1], enforce_unique=False), VRF(name='VRF 4', rd='65000:400', tenant=tenants[1], enforce_unique=True), VRF(name='VRF 5', rd='65000:500', tenant=tenants[2], enforce_unique=True), VRF(name='VRF 6', rd='65000:600', tenant=tenants[2], enforce_unique=True))
        VRF.objects.bulk_create(vrfs)
        vrfs[0].import_targets.add(route_targets[0])
        vrfs[0].export_targets.add(route_targets[0])
        vrfs[1].import_targets.add(route_targets[1])
        vrfs[1].export_targets.add(route_targets[1])
        vrfs[2].import_targets.add(route_targets[2])
        vrfs[2].export_targets.add(route_targets[2])

    def test_name(self):
        if False:
            for i in range(10):
                print('nop')
        params = {'name': ['VRF 1', 'VRF 2']}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 2)

    def test_rd(self):
        if False:
            while True:
                i = 10
        params = {'rd': ['65000:100', '65000:200']}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 2)

    def test_enforce_unique(self):
        if False:
            return 10
        params = {'enforce_unique': 'true'}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 3)
        params = {'enforce_unique': 'false'}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 3)

    def test_import_target(self):
        if False:
            for i in range(10):
                print('nop')
        route_targets = RouteTarget.objects.all()[:2]
        params = {'import_target_id': [route_targets[0].pk, route_targets[1].pk]}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 2)
        params = {'import_target': [route_targets[0].name, route_targets[1].name]}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 2)

    def test_export_target(self):
        if False:
            i = 10
            return i + 15
        route_targets = RouteTarget.objects.all()[:2]
        params = {'export_target_id': [route_targets[0].pk, route_targets[1].pk]}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 2)
        params = {'export_target': [route_targets[0].name, route_targets[1].name]}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 2)

    def test_tenant(self):
        if False:
            for i in range(10):
                print('nop')
        tenants = Tenant.objects.all()[:2]
        params = {'tenant_id': [tenants[0].pk, tenants[1].pk]}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 4)
        params = {'tenant': [tenants[0].slug, tenants[1].slug]}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 4)

    def test_tenant_group(self):
        if False:
            print('Hello World!')
        tenant_groups = TenantGroup.objects.all()[:2]
        params = {'tenant_group_id': [tenant_groups[0].pk, tenant_groups[1].pk]}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 4)
        params = {'tenant_group': [tenant_groups[0].slug, tenant_groups[1].slug]}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 4)

    def test_description(self):
        if False:
            i = 10
            return i + 15
        params = {'description': ['foobar1', 'foobar2']}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 2)

class RouteTargetTestCase(TestCase, ChangeLoggedFilterSetTests):
    queryset = RouteTarget.objects.all()
    filterset = RouteTargetFilterSet

    @classmethod
    def setUpTestData(cls):
        if False:
            for i in range(10):
                print('nop')
        tenant_groups = (TenantGroup(name='Tenant group 1', slug='tenant-group-1'), TenantGroup(name='Tenant group 2', slug='tenant-group-2'), TenantGroup(name='Tenant group 3', slug='tenant-group-3'))
        for tenantgroup in tenant_groups:
            tenantgroup.save()
        tenants = (Tenant(name='Tenant 1', slug='tenant-1', group=tenant_groups[0]), Tenant(name='Tenant 2', slug='tenant-2', group=tenant_groups[1]), Tenant(name='Tenant 3', slug='tenant-3', group=tenant_groups[2]))
        Tenant.objects.bulk_create(tenants)
        route_targets = (RouteTarget(name='65000:1001', tenant=tenants[0], description='foobar1'), RouteTarget(name='65000:1002', tenant=tenants[0], description='foobar2'), RouteTarget(name='65000:1003', tenant=tenants[0]), RouteTarget(name='65000:1004', tenant=tenants[0]), RouteTarget(name='65000:2001', tenant=tenants[1]), RouteTarget(name='65000:2002', tenant=tenants[1]), RouteTarget(name='65000:2003', tenant=tenants[1]), RouteTarget(name='65000:2004', tenant=tenants[1]), RouteTarget(name='65000:3001', tenant=tenants[2]), RouteTarget(name='65000:3002', tenant=tenants[2]), RouteTarget(name='65000:3003', tenant=tenants[2]), RouteTarget(name='65000:3004', tenant=tenants[2]))
        RouteTarget.objects.bulk_create(route_targets)
        vrfs = (VRF(name='VRF 1', rd='65000:100'), VRF(name='VRF 2', rd='65000:200'), VRF(name='VRF 3', rd='65000:300'))
        VRF.objects.bulk_create(vrfs)
        vrfs[0].import_targets.add(route_targets[0], route_targets[1])
        vrfs[0].export_targets.add(route_targets[2], route_targets[3])
        vrfs[1].import_targets.add(route_targets[4], route_targets[5])
        vrfs[1].export_targets.add(route_targets[6], route_targets[7])

    def test_name(self):
        if False:
            while True:
                i = 10
        params = {'name': ['65000:1001', '65000:1002', '65000:1003']}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 3)

    def test_importing_vrf(self):
        if False:
            for i in range(10):
                print('nop')
        vrfs = VRF.objects.all()[:2]
        params = {'importing_vrf_id': [vrfs[0].pk, vrfs[1].pk]}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 4)
        params = {'importing_vrf': [vrfs[0].rd, vrfs[1].rd]}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 4)

    def test_exporting_vrf(self):
        if False:
            print('Hello World!')
        vrfs = VRF.objects.all()[:2]
        params = {'exporting_vrf_id': [vrfs[0].pk, vrfs[1].pk]}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 4)
        params = {'exporting_vrf': [vrfs[0].rd, vrfs[1].rd]}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 4)

    def test_tenant(self):
        if False:
            while True:
                i = 10
        tenants = Tenant.objects.all()[:2]
        params = {'tenant_id': [tenants[0].pk, tenants[1].pk]}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 8)
        params = {'tenant': [tenants[0].slug, tenants[1].slug]}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 8)

    def test_tenant_group(self):
        if False:
            i = 10
            return i + 15
        tenant_groups = TenantGroup.objects.all()[:2]
        params = {'tenant_group_id': [tenant_groups[0].pk, tenant_groups[1].pk]}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 8)
        params = {'tenant_group': [tenant_groups[0].slug, tenant_groups[1].slug]}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 8)

    def test_description(self):
        if False:
            for i in range(10):
                print('nop')
        params = {'description': ['foobar1', 'foobar2']}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 2)

class RIRTestCase(TestCase, ChangeLoggedFilterSetTests):
    queryset = RIR.objects.all()
    filterset = RIRFilterSet

    @classmethod
    def setUpTestData(cls):
        if False:
            return 10
        rirs = (RIR(name='RIR 1', slug='rir-1', is_private=False, description='A'), RIR(name='RIR 2', slug='rir-2', is_private=False, description='B'), RIR(name='RIR 3', slug='rir-3', is_private=False, description='C'), RIR(name='RIR 4', slug='rir-4', is_private=True, description='D'), RIR(name='RIR 5', slug='rir-5', is_private=True, description='E'), RIR(name='RIR 6', slug='rir-6', is_private=True, description='F'))
        RIR.objects.bulk_create(rirs)

    def test_name(self):
        if False:
            while True:
                i = 10
        params = {'name': ['RIR 1', 'RIR 2']}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 2)

    def test_slug(self):
        if False:
            while True:
                i = 10
        params = {'slug': ['rir-1', 'rir-2']}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 2)

    def test_description(self):
        if False:
            while True:
                i = 10
        params = {'description': ['A', 'B']}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 2)

    def test_is_private(self):
        if False:
            i = 10
            return i + 15
        params = {'is_private': 'true'}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 3)
        params = {'is_private': 'false'}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 3)

class AggregateTestCase(TestCase, ChangeLoggedFilterSetTests):
    queryset = Aggregate.objects.all()
    filterset = AggregateFilterSet

    @classmethod
    def setUpTestData(cls):
        if False:
            for i in range(10):
                print('nop')
        rirs = (RIR(name='RIR 1', slug='rir-1'), RIR(name='RIR 2', slug='rir-2'), RIR(name='RIR 3', slug='rir-3'))
        RIR.objects.bulk_create(rirs)
        tenant_groups = (TenantGroup(name='Tenant group 1', slug='tenant-group-1'), TenantGroup(name='Tenant group 2', slug='tenant-group-2'), TenantGroup(name='Tenant group 3', slug='tenant-group-3'))
        for tenantgroup in tenant_groups:
            tenantgroup.save()
        tenants = (Tenant(name='Tenant 1', slug='tenant-1', group=tenant_groups[0]), Tenant(name='Tenant 2', slug='tenant-2', group=tenant_groups[1]), Tenant(name='Tenant 3', slug='tenant-3', group=tenant_groups[2]))
        Tenant.objects.bulk_create(tenants)
        aggregates = (Aggregate(prefix='10.1.0.0/16', rir=rirs[0], tenant=tenants[0], date_added='2020-01-01', description='foobar1'), Aggregate(prefix='10.2.0.0/16', rir=rirs[0], tenant=tenants[1], date_added='2020-01-02', description='foobar2'), Aggregate(prefix='10.3.0.0/16', rir=rirs[1], tenant=tenants[2], date_added='2020-01-03'), Aggregate(prefix='2001:db8:1::/48', rir=rirs[1], tenant=tenants[0], date_added='2020-01-04'), Aggregate(prefix='2001:db8:2::/48', rir=rirs[2], tenant=tenants[1], date_added='2020-01-05'), Aggregate(prefix='2001:db8:3::/48', rir=rirs[2], tenant=tenants[2], date_added='2020-01-06'))
        Aggregate.objects.bulk_create(aggregates)

    def test_family(self):
        if False:
            print('Hello World!')
        params = {'family': '4'}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 3)

    def test_date_added(self):
        if False:
            for i in range(10):
                print('nop')
        params = {'date_added': ['2020-01-01', '2020-01-02']}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 2)

    def test_description(self):
        if False:
            print('Hello World!')
        params = {'description': ['foobar1', 'foobar2']}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 2)

    def test_prefix(self):
        if False:
            for i in range(10):
                print('nop')
        params = {'prefix': '10.1.0.0/16'}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 1)

    def test_rir(self):
        if False:
            for i in range(10):
                print('nop')
        rirs = RIR.objects.all()[:2]
        params = {'rir_id': [rirs[0].pk, rirs[1].pk]}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 4)
        params = {'rir': [rirs[0].slug, rirs[1].slug]}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 4)

    def test_tenant(self):
        if False:
            for i in range(10):
                print('nop')
        tenants = Tenant.objects.all()[:2]
        params = {'tenant_id': [tenants[0].pk, tenants[1].pk]}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 4)
        params = {'tenant': [tenants[0].slug, tenants[1].slug]}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 4)

    def test_tenant_group(self):
        if False:
            while True:
                i = 10
        tenant_groups = TenantGroup.objects.all()[:2]
        params = {'tenant_group_id': [tenant_groups[0].pk, tenant_groups[1].pk]}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 4)
        params = {'tenant_group': [tenant_groups[0].slug, tenant_groups[1].slug]}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 4)

class RoleTestCase(TestCase, ChangeLoggedFilterSetTests):
    queryset = Role.objects.all()
    filterset = RoleFilterSet

    @classmethod
    def setUpTestData(cls):
        if False:
            while True:
                i = 10
        roles = (Role(name='Role 1', slug='role-1', description='foobar1'), Role(name='Role 2', slug='role-2', description='foobar2'), Role(name='Role 3', slug='role-3'))
        Role.objects.bulk_create(roles)

    def test_name(self):
        if False:
            for i in range(10):
                print('nop')
        params = {'name': ['Role 1', 'Role 2']}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 2)

    def test_slug(self):
        if False:
            return 10
        params = {'slug': ['role-1', 'role-2']}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 2)

    def test_description(self):
        if False:
            for i in range(10):
                print('nop')
        params = {'description': ['foobar1', 'foobar2']}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 2)

class PrefixTestCase(TestCase, ChangeLoggedFilterSetTests):
    queryset = Prefix.objects.all()
    filterset = PrefixFilterSet

    @classmethod
    def setUpTestData(cls):
        if False:
            return 10
        regions = (Region(name='Test Region 1', slug='test-region-1'), Region(name='Test Region 2', slug='test-region-2'), Region(name='Test Region 3', slug='test-region-3'))
        for r in regions:
            r.save()
        site_groups = (SiteGroup(name='Site Group 1', slug='site-group-1'), SiteGroup(name='Site Group 2', slug='site-group-2'), SiteGroup(name='Site Group 3', slug='site-group-3'))
        for site_group in site_groups:
            site_group.save()
        sites = (Site(name='Test Site 1', slug='test-site-1', region=regions[0], group=site_groups[0]), Site(name='Test Site 2', slug='test-site-2', region=regions[1], group=site_groups[1]), Site(name='Test Site 3', slug='test-site-3', region=regions[2], group=site_groups[2]))
        Site.objects.bulk_create(sites)
        route_targets = (RouteTarget(name='65000:100'), RouteTarget(name='65000:200'), RouteTarget(name='65000:300'))
        RouteTarget.objects.bulk_create(route_targets)
        vrfs = (VRF(name='VRF 1', rd='65000:100'), VRF(name='VRF 2', rd='65000:200'), VRF(name='VRF 3', rd='65000:300'))
        VRF.objects.bulk_create(vrfs)
        vrfs[0].import_targets.add(route_targets[0], route_targets[1], route_targets[2])
        vrfs[1].export_targets.add(route_targets[1])
        vrfs[2].export_targets.add(route_targets[2])
        vlans = (VLAN(vid=1, name='VLAN 1'), VLAN(vid=2, name='VLAN 2'), VLAN(vid=3, name='VLAN 3'))
        VLAN.objects.bulk_create(vlans)
        roles = (Role(name='Role 1', slug='role-1'), Role(name='Role 2', slug='role-2'), Role(name='Role 3', slug='role-3'))
        Role.objects.bulk_create(roles)
        tenant_groups = (TenantGroup(name='Tenant group 1', slug='tenant-group-1'), TenantGroup(name='Tenant group 2', slug='tenant-group-2'), TenantGroup(name='Tenant group 3', slug='tenant-group-3'))
        for tenantgroup in tenant_groups:
            tenantgroup.save()
        tenants = (Tenant(name='Tenant 1', slug='tenant-1', group=tenant_groups[0]), Tenant(name='Tenant 2', slug='tenant-2', group=tenant_groups[1]), Tenant(name='Tenant 3', slug='tenant-3', group=tenant_groups[2]))
        Tenant.objects.bulk_create(tenants)
        prefixes = (Prefix(prefix='10.0.0.0/24', tenant=None, site=None, vrf=None, vlan=None, role=None, is_pool=True, mark_utilized=True, description='foobar1'), Prefix(prefix='10.0.1.0/24', tenant=tenants[0], site=sites[0], vrf=vrfs[0], vlan=vlans[0], role=roles[0], description='foobar2'), Prefix(prefix='10.0.2.0/24', tenant=tenants[1], site=sites[1], vrf=vrfs[1], vlan=vlans[1], role=roles[1], status=PrefixStatusChoices.STATUS_DEPRECATED), Prefix(prefix='10.0.3.0/24', tenant=tenants[2], site=sites[2], vrf=vrfs[2], vlan=vlans[2], role=roles[2], status=PrefixStatusChoices.STATUS_RESERVED), Prefix(prefix='2001:db8::/64', tenant=None, site=None, vrf=None, vlan=None, role=None, is_pool=True, mark_utilized=True), Prefix(prefix='2001:db8:0:1::/64', tenant=tenants[0], site=sites[0], vrf=vrfs[0], vlan=vlans[0], role=roles[0]), Prefix(prefix='2001:db8:0:2::/64', tenant=tenants[1], site=sites[1], vrf=vrfs[1], vlan=vlans[1], role=roles[1], status=PrefixStatusChoices.STATUS_DEPRECATED), Prefix(prefix='2001:db8:0:3::/64', tenant=tenants[2], site=sites[2], vrf=vrfs[2], vlan=vlans[2], role=roles[2], status=PrefixStatusChoices.STATUS_RESERVED), Prefix(prefix='10.0.0.0/16'), Prefix(prefix='2001:db8::/32'))
        for prefix in prefixes:
            prefix.save()

    def test_family(self):
        if False:
            print('Hello World!')
        params = {'family': '6'}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 5)

    def test_prefix(self):
        if False:
            print('Hello World!')
        prefixes = Prefix.objects.all()[:2]
        params = {'prefix': [prefixes[0].prefix, prefixes[1].prefix]}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 2)

    def test_is_pool(self):
        if False:
            while True:
                i = 10
        params = {'is_pool': 'true'}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 2)
        params = {'is_pool': 'false'}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 8)

    def test_mark_utilized(self):
        if False:
            print('Hello World!')
        params = {'mark_utilized': 'true'}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 2)
        params = {'mark_utilized': 'false'}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 8)

    def test_within(self):
        if False:
            i = 10
            return i + 15
        params = {'within': '10.0.0.0/16'}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 4)

    def test_within_include(self):
        if False:
            for i in range(10):
                print('nop')
        params = {'within_include': '10.0.0.0/16'}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 5)

    def test_contains(self):
        if False:
            i = 10
            return i + 15
        params = {'contains': '10.0.1.0/24'}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 2)
        params = {'contains': '2001:db8:0:1::/64'}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 2)

    def test_depth(self):
        if False:
            print('Hello World!')
        params = {'depth': '0'}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 8)
        params = {'depth__gt': '0'}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 2)

    def test_children(self):
        if False:
            for i in range(10):
                print('nop')
        params = {'children': '0'}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 8)
        params = {'children__gt': '0'}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 2)

    def test_mask_length(self):
        if False:
            return 10
        params = {'mask_length': [24]}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 4)
        params = {'mask_length__gte': 32}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 5)
        params = {'mask_length__lte': 24}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 5)

    def test_vrf(self):
        if False:
            print('Hello World!')
        vrfs = VRF.objects.all()[:2]
        params = {'vrf_id': [vrfs[0].pk, vrfs[1].pk]}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 4)
        params = {'vrf': [vrfs[0].rd, vrfs[1].rd]}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 4)

    def test_present_in_vrf(self):
        if False:
            i = 10
            return i + 15
        vrf1 = VRF.objects.get(name='VRF 1')
        vrf2 = VRF.objects.get(name='VRF 2')
        self.assertEqual(self.filterset({'present_in_vrf_id': vrf1.pk}, self.queryset).qs.count(), 6)
        self.assertEqual(self.filterset({'present_in_vrf_id': vrf2.pk}, self.queryset).qs.count(), 2)
        self.assertEqual(self.filterset({'present_in_vrf': vrf1.rd}, self.queryset).qs.count(), 6)
        self.assertEqual(self.filterset({'present_in_vrf': vrf2.rd}, self.queryset).qs.count(), 2)

    def test_region(self):
        if False:
            i = 10
            return i + 15
        regions = Region.objects.all()[:2]
        params = {'region_id': [regions[0].pk, regions[1].pk]}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 4)
        params = {'region': [regions[0].slug, regions[1].slug]}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 4)

    def test_site_group(self):
        if False:
            return 10
        site_groups = SiteGroup.objects.all()[:2]
        params = {'site_group_id': [site_groups[0].pk, site_groups[1].pk]}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 4)
        params = {'site_group': [site_groups[0].slug, site_groups[1].slug]}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 4)

    def test_site(self):
        if False:
            print('Hello World!')
        sites = Site.objects.all()[:2]
        params = {'site_id': [sites[0].pk, sites[1].pk]}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 4)
        params = {'site': [sites[0].slug, sites[1].slug]}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 4)

    def test_vlan(self):
        if False:
            i = 10
            return i + 15
        vlans = VLAN.objects.all()[:2]
        params = {'vlan_id': [vlans[0].pk, vlans[1].pk]}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 4)
        params = {'vlan_vid': vlans[0].vid}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 2)

    def test_role(self):
        if False:
            print('Hello World!')
        roles = Role.objects.all()[:2]
        params = {'role_id': [roles[0].pk, roles[1].pk]}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 4)
        params = {'role': [roles[0].slug, roles[1].slug]}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 4)

    def test_status(self):
        if False:
            for i in range(10):
                print('nop')
        params = {'status': [PrefixStatusChoices.STATUS_DEPRECATED, PrefixStatusChoices.STATUS_RESERVED]}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 4)

    def test_tenant(self):
        if False:
            while True:
                i = 10
        tenants = Tenant.objects.all()[:2]
        params = {'tenant_id': [tenants[0].pk, tenants[1].pk]}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 4)
        params = {'tenant': [tenants[0].slug, tenants[1].slug]}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 4)

    def test_tenant_group(self):
        if False:
            print('Hello World!')
        tenant_groups = TenantGroup.objects.all()[:2]
        params = {'tenant_group_id': [tenant_groups[0].pk, tenant_groups[1].pk]}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 4)
        params = {'tenant_group': [tenant_groups[0].slug, tenant_groups[1].slug]}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 4)

    def test_description(self):
        if False:
            return 10
        params = {'description': ['foobar1', 'foobar2']}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 2)

class IPRangeTestCase(TestCase, ChangeLoggedFilterSetTests):
    queryset = IPRange.objects.all()
    filterset = IPRangeFilterSet

    @classmethod
    def setUpTestData(cls):
        if False:
            print('Hello World!')
        vrfs = (VRF(name='VRF 1', rd='65000:100'), VRF(name='VRF 2', rd='65000:200'), VRF(name='VRF 3', rd='65000:300'))
        VRF.objects.bulk_create(vrfs)
        roles = (Role(name='Role 1', slug='role-1'), Role(name='Role 2', slug='role-2'), Role(name='Role 3', slug='role-3'))
        Role.objects.bulk_create(roles)
        tenant_groups = (TenantGroup(name='Tenant group 1', slug='tenant-group-1'), TenantGroup(name='Tenant group 2', slug='tenant-group-2'), TenantGroup(name='Tenant group 3', slug='tenant-group-3'))
        for tenantgroup in tenant_groups:
            tenantgroup.save()
        tenants = (Tenant(name='Tenant 1', slug='tenant-1', group=tenant_groups[0]), Tenant(name='Tenant 2', slug='tenant-2', group=tenant_groups[1]), Tenant(name='Tenant 3', slug='tenant-3', group=tenant_groups[2]))
        Tenant.objects.bulk_create(tenants)
        ip_ranges = (IPRange(start_address='10.0.1.100/24', end_address='10.0.1.199/24', size=100, vrf=None, tenant=None, role=None, status=IPRangeStatusChoices.STATUS_ACTIVE, description='foobar1'), IPRange(start_address='10.0.2.100/24', end_address='10.0.2.199/24', size=100, vrf=vrfs[0], tenant=tenants[0], role=roles[0], status=IPRangeStatusChoices.STATUS_ACTIVE, description='foobar2'), IPRange(start_address='10.0.3.100/24', end_address='10.0.3.199/24', size=100, vrf=vrfs[1], tenant=tenants[1], role=roles[1], status=IPRangeStatusChoices.STATUS_DEPRECATED), IPRange(start_address='10.0.4.100/24', end_address='10.0.4.199/24', size=100, vrf=vrfs[2], tenant=tenants[2], role=roles[2], status=IPRangeStatusChoices.STATUS_RESERVED), IPRange(start_address='2001:db8:0:1::1/64', end_address='2001:db8:0:1::100/64', size=100, vrf=None, tenant=None, role=None, status=IPRangeStatusChoices.STATUS_ACTIVE), IPRange(start_address='2001:db8:0:2::1/64', end_address='2001:db8:0:2::100/64', size=100, vrf=vrfs[0], tenant=tenants[0], role=roles[0], status=IPRangeStatusChoices.STATUS_ACTIVE), IPRange(start_address='2001:db8:0:3::1/64', end_address='2001:db8:0:3::100/64', size=100, vrf=vrfs[1], tenant=tenants[1], role=roles[1], status=IPRangeStatusChoices.STATUS_DEPRECATED), IPRange(start_address='2001:db8:0:4::1/64', end_address='2001:db8:0:4::100/64', size=100, vrf=vrfs[2], tenant=tenants[2], role=roles[2], status=IPRangeStatusChoices.STATUS_RESERVED))
        IPRange.objects.bulk_create(ip_ranges)

    def test_family(self):
        if False:
            for i in range(10):
                print('nop')
        params = {'family': '6'}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 4)

    def test_start_address(self):
        if False:
            return 10
        params = {'start_address': ['10.0.1.100', '10.0.2.100']}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 2)

    def test_end_address(self):
        if False:
            print('Hello World!')
        params = {'end_address': ['10.0.1.199', '10.0.2.199']}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 2)

    def test_contains(self):
        if False:
            print('Hello World!')
        params = {'contains': '10.0.1.150/24'}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 1)
        params = {'contains': '2001:db8:0:1::50/64'}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 1)

    def test_vrf(self):
        if False:
            print('Hello World!')
        vrfs = VRF.objects.all()[:2]
        params = {'vrf_id': [vrfs[0].pk, vrfs[1].pk]}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 4)
        params = {'vrf': [vrfs[0].rd, vrfs[1].rd]}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 4)

    def test_role(self):
        if False:
            while True:
                i = 10
        roles = Role.objects.all()[:2]
        params = {'role_id': [roles[0].pk, roles[1].pk]}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 4)
        params = {'role': [roles[0].slug, roles[1].slug]}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 4)

    def test_status(self):
        if False:
            return 10
        params = {'status': [PrefixStatusChoices.STATUS_DEPRECATED, PrefixStatusChoices.STATUS_RESERVED]}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 4)

    def test_tenant(self):
        if False:
            print('Hello World!')
        tenants = Tenant.objects.all()[:2]
        params = {'tenant_id': [tenants[0].pk, tenants[1].pk]}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 4)
        params = {'tenant': [tenants[0].slug, tenants[1].slug]}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 4)

    def test_tenant_group(self):
        if False:
            while True:
                i = 10
        tenant_groups = TenantGroup.objects.all()[:2]
        params = {'tenant_group_id': [tenant_groups[0].pk, tenant_groups[1].pk]}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 4)
        params = {'tenant_group': [tenant_groups[0].slug, tenant_groups[1].slug]}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 4)

    def test_description(self):
        if False:
            i = 10
            return i + 15
        params = {'description': ['foobar1', 'foobar2']}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 2)

    def test_parent(self):
        if False:
            for i in range(10):
                print('nop')
        params = {'parent': ['10.0.1.0/24', '10.0.2.0/24']}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 2)
        params = {'parent': ['10.0.1.0/25']}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 0)

class IPAddressTestCase(TestCase, ChangeLoggedFilterSetTests):
    queryset = IPAddress.objects.all()
    filterset = IPAddressFilterSet

    @classmethod
    def setUpTestData(cls):
        if False:
            return 10
        vrfs = (VRF(name='VRF 1', rd='65000:100'), VRF(name='VRF 2', rd='65000:200'), VRF(name='VRF 3', rd='65000:300'))
        VRF.objects.bulk_create(vrfs)
        site = Site.objects.create(name='Site 1', slug='site-1')
        manufacturer = Manufacturer.objects.create(name='Manufacturer 1', slug='manufacturer-1')
        device_type = DeviceType.objects.create(manufacturer=manufacturer, model='Device Type 1')
        role = DeviceRole.objects.create(name='Device Role 1', slug='device-role-1')
        devices = (Device(device_type=device_type, name='Device 1', site=site, role=role), Device(device_type=device_type, name='Device 2', site=site, role=role), Device(device_type=device_type, name='Device 3', site=site, role=role))
        Device.objects.bulk_create(devices)
        interfaces = (Interface(device=devices[0], name='Interface 1'), Interface(device=devices[1], name='Interface 2'), Interface(device=devices[2], name='Interface 3'))
        Interface.objects.bulk_create(interfaces)
        clustertype = ClusterType.objects.create(name='Cluster Type 1', slug='cluster-type-1')
        cluster = Cluster.objects.create(type=clustertype, name='Cluster 1')
        virtual_machines = (VirtualMachine(name='Virtual Machine 1', cluster=cluster), VirtualMachine(name='Virtual Machine 2', cluster=cluster), VirtualMachine(name='Virtual Machine 3', cluster=cluster))
        VirtualMachine.objects.bulk_create(virtual_machines)
        vminterfaces = (VMInterface(virtual_machine=virtual_machines[0], name='Interface 1'), VMInterface(virtual_machine=virtual_machines[1], name='Interface 2'), VMInterface(virtual_machine=virtual_machines[2], name='Interface 3'))
        VMInterface.objects.bulk_create(vminterfaces)
        fhrp_groups = (FHRPGroup(protocol=FHRPGroupProtocolChoices.PROTOCOL_VRRP2, group_id=101), FHRPGroup(protocol=FHRPGroupProtocolChoices.PROTOCOL_VRRP2, group_id=102))
        FHRPGroup.objects.bulk_create(fhrp_groups)
        tenant_groups = (TenantGroup(name='Tenant group 1', slug='tenant-group-1'), TenantGroup(name='Tenant group 2', slug='tenant-group-2'), TenantGroup(name='Tenant group 3', slug='tenant-group-3'))
        for tenantgroup in tenant_groups:
            tenantgroup.save()
        tenants = (Tenant(name='Tenant 1', slug='tenant-1', group=tenant_groups[0]), Tenant(name='Tenant 2', slug='tenant-2', group=tenant_groups[1]), Tenant(name='Tenant 3', slug='tenant-3', group=tenant_groups[2]))
        Tenant.objects.bulk_create(tenants)
        ipaddresses = (IPAddress(address='10.0.0.1/24', tenant=None, vrf=None, assigned_object=None, status=IPAddressStatusChoices.STATUS_ACTIVE, dns_name='ipaddress-a', description='foobar1'), IPAddress(address='10.0.0.2/24', tenant=tenants[0], vrf=vrfs[0], assigned_object=interfaces[0], status=IPAddressStatusChoices.STATUS_ACTIVE, dns_name='ipaddress-b'), IPAddress(address='10.0.0.3/24', tenant=tenants[1], vrf=vrfs[1], assigned_object=interfaces[1], status=IPAddressStatusChoices.STATUS_RESERVED, role=IPAddressRoleChoices.ROLE_VIP, dns_name='ipaddress-c'), IPAddress(address='10.0.0.4/24', tenant=tenants[2], vrf=vrfs[2], assigned_object=interfaces[2], status=IPAddressStatusChoices.STATUS_DEPRECATED, role=IPAddressRoleChoices.ROLE_SECONDARY, dns_name='ipaddress-d'), IPAddress(address='10.0.0.5/24', tenant=None, vrf=None, assigned_object=fhrp_groups[0], status=IPAddressStatusChoices.STATUS_ACTIVE), IPAddress(address='10.0.0.1/25', tenant=None, vrf=None, assigned_object=None, status=IPAddressStatusChoices.STATUS_ACTIVE), IPAddress(address='2001:db8::1/64', tenant=None, vrf=None, assigned_object=None, status=IPAddressStatusChoices.STATUS_ACTIVE, dns_name='ipaddress-a', description='foobar2'), IPAddress(address='2001:db8::2/64', tenant=tenants[0], vrf=vrfs[0], assigned_object=vminterfaces[0], status=IPAddressStatusChoices.STATUS_ACTIVE, dns_name='ipaddress-b'), IPAddress(address='2001:db8::3/64', tenant=tenants[1], vrf=vrfs[1], assigned_object=vminterfaces[1], status=IPAddressStatusChoices.STATUS_RESERVED, role=IPAddressRoleChoices.ROLE_VIP, dns_name='ipaddress-c'), IPAddress(address='2001:db8::4/64', tenant=tenants[2], vrf=vrfs[2], assigned_object=vminterfaces[2], status=IPAddressStatusChoices.STATUS_DEPRECATED, role=IPAddressRoleChoices.ROLE_SECONDARY, dns_name='ipaddress-d'), IPAddress(address='2001:db8::5/64', tenant=None, vrf=None, assigned_object=fhrp_groups[1], status=IPAddressStatusChoices.STATUS_ACTIVE), IPAddress(address='2001:db8::1/65', tenant=None, vrf=None, assigned_object=None, status=IPAddressStatusChoices.STATUS_ACTIVE))
        IPAddress.objects.bulk_create(ipaddresses)

    def test_family(self):
        if False:
            return 10
        params = {'family': '4'}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 6)
        params = {'family': '6'}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 6)

    def test_dns_name(self):
        if False:
            i = 10
            return i + 15
        params = {'dns_name': ['ipaddress-a', 'ipaddress-b']}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 4)

    def test_description(self):
        if False:
            for i in range(10):
                print('nop')
        params = {'description': ['foobar1', 'foobar2']}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 2)

    def test_parent(self):
        if False:
            for i in range(10):
                print('nop')
        params = {'parent': ['10.0.0.0/30', '2001:db8::/126']}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 8)

    def test_filter_address(self):
        if False:
            for i in range(10):
                print('nop')
        params = {'address': ['10.0.0.1/24']}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 1)
        params = {'address': ['10.0.0.1']}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 2)
        params = {'address': ['10.0.0.1/24', '10.0.0.1/25']}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 2)
        params = {'address': ['2001:db8::1/64']}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 1)
        params = {'address': ['2001:db8::1']}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 2)
        params = {'address': ['2001:db8::1/64', '2001:db8::1/65']}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 2)
        params = {'address': ['24']}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 0)
        params = {'address': ['10.0.0.1/255.255.255.0']}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 1)
        params = {'address': ['10.0.0.1/255.255.255.0', '10.0.0.1/25']}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 2)
        params = {'address': ['/24']}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 0)
        params = {'address': ['10.0.0.1/255.255.999.0']}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 0)
        params = {'address': ['10.0.0.1', '/24', '10.0.0.10/24']}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 2)

    def test_mask_length(self):
        if False:
            while True:
                i = 10
        params = {'mask_length': [24]}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 5)
        params = {'mask_length__gte': 64}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 6)
        params = {'mask_length__lte': 25}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 6)

    def test_vrf(self):
        if False:
            while True:
                i = 10
        vrfs = VRF.objects.all()[:2]
        params = {'vrf_id': [vrfs[0].pk, vrfs[1].pk]}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 4)
        params = {'vrf': [vrfs[0].rd, vrfs[1].rd]}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 4)

    def test_device(self):
        if False:
            print('Hello World!')
        devices = Device.objects.all()[:2]
        params = {'device_id': [devices[0].pk, devices[1].pk]}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 2)
        params = {'device': [devices[0].name, devices[1].name]}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 2)

    def test_virtual_machine(self):
        if False:
            for i in range(10):
                print('nop')
        vms = VirtualMachine.objects.all()[:2]
        params = {'virtual_machine_id': [vms[0].pk, vms[1].pk]}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 2)
        params = {'virtual_machine': [vms[0].name, vms[1].name]}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 2)

    def test_interface(self):
        if False:
            print('Hello World!')
        interfaces = Interface.objects.all()[:2]
        params = {'interface_id': [interfaces[0].pk, interfaces[1].pk]}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 2)
        params = {'interface': ['Interface 1', 'Interface 2']}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 2)

    def test_vminterface(self):
        if False:
            i = 10
            return i + 15
        vminterfaces = VMInterface.objects.all()[:2]
        params = {'vminterface_id': [vminterfaces[0].pk, vminterfaces[1].pk]}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 2)
        params = {'vminterface': ['Interface 1', 'Interface 2']}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 2)

    def test_fhrpgroup(self):
        if False:
            i = 10
            return i + 15
        fhrp_groups = FHRPGroup.objects.all()[:2]
        params = {'fhrpgroup_id': [fhrp_groups[0].pk, fhrp_groups[1].pk]}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 2)

    def test_assigned(self):
        if False:
            for i in range(10):
                print('nop')
        params = {'assigned': 'true'}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 8)
        params = {'assigned': 'false'}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 4)

    def test_assigned_to_interface(self):
        if False:
            while True:
                i = 10
        params = {'assigned_to_interface': 'true'}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 6)
        params = {'assigned_to_interface': 'false'}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 6)

    def test_status(self):
        if False:
            while True:
                i = 10
        params = {'status': [PrefixStatusChoices.STATUS_DEPRECATED, PrefixStatusChoices.STATUS_RESERVED]}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 4)

    def test_role(self):
        if False:
            return 10
        params = {'role': [IPAddressRoleChoices.ROLE_SECONDARY, IPAddressRoleChoices.ROLE_VIP]}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 4)

    def test_tenant(self):
        if False:
            print('Hello World!')
        tenants = Tenant.objects.all()[:2]
        params = {'tenant_id': [tenants[0].pk, tenants[1].pk]}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 4)
        params = {'tenant': [tenants[0].slug, tenants[1].slug]}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 4)

    def test_tenant_group(self):
        if False:
            print('Hello World!')
        tenant_groups = TenantGroup.objects.all()[:2]
        params = {'tenant_group_id': [tenant_groups[0].pk, tenant_groups[1].pk]}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 4)
        params = {'tenant_group': [tenant_groups[0].slug, tenant_groups[1].slug]}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 4)

class FHRPGroupTestCase(TestCase, ChangeLoggedFilterSetTests):
    queryset = FHRPGroup.objects.all()
    filterset = FHRPGroupFilterSet

    @classmethod
    def setUpTestData(cls):
        if False:
            print('Hello World!')
        ip_addresses = (IPAddress(address=IPNetwork('192.168.1.1/24')), IPAddress(address=IPNetwork('192.168.2.1/24')), IPAddress(address=IPNetwork('192.168.3.1/24')))
        IPAddress.objects.bulk_create(ip_addresses)
        fhrp_groups = (FHRPGroup(protocol=FHRPGroupProtocolChoices.PROTOCOL_VRRP2, group_id=10, auth_type=FHRPGroupAuthTypeChoices.AUTHENTICATION_PLAINTEXT, auth_key='foo123'), FHRPGroup(protocol=FHRPGroupProtocolChoices.PROTOCOL_VRRP3, group_id=20, auth_type=FHRPGroupAuthTypeChoices.AUTHENTICATION_MD5, auth_key='bar456', name='bar123'), FHRPGroup(protocol=FHRPGroupProtocolChoices.PROTOCOL_HSRP, group_id=30))
        FHRPGroup.objects.bulk_create(fhrp_groups)
        fhrp_groups[0].ip_addresses.set([ip_addresses[0]])
        fhrp_groups[1].ip_addresses.set([ip_addresses[1]])
        fhrp_groups[2].ip_addresses.set([ip_addresses[2]])

    def test_protocol(self):
        if False:
            for i in range(10):
                print('nop')
        params = {'protocol': [FHRPGroupProtocolChoices.PROTOCOL_VRRP2, FHRPGroupProtocolChoices.PROTOCOL_VRRP3]}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 2)

    def test_group_id(self):
        if False:
            print('Hello World!')
        params = {'group_id': [10, 20]}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 2)

    def test_auth_type(self):
        if False:
            for i in range(10):
                print('nop')
        params = {'auth_type': [FHRPGroupAuthTypeChoices.AUTHENTICATION_PLAINTEXT, FHRPGroupAuthTypeChoices.AUTHENTICATION_MD5]}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 2)

    def test_auth_key(self):
        if False:
            i = 10
            return i + 15
        params = {'auth_key': ['foo123', 'bar456']}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 2)

    def test_name(self):
        if False:
            i = 10
            return i + 15
        params = {'name': ['bar123']}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 1)

    def test_related_ip(self):
        if False:
            i = 10
            return i + 15
        ipaddresses = (IPAddress.objects.create(address=IPNetwork('192.168.1.2/24')), IPAddress.objects.create(address=IPNetwork('192.168.2.2/24')))
        params = {'related_ip': [ipaddresses[0].pk, ipaddresses[1].pk]}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 2)

class FHRPGroupAssignmentTestCase(TestCase, ChangeLoggedFilterSetTests):
    queryset = FHRPGroupAssignment.objects.all()
    filterset = FHRPGroupAssignmentFilterSet

    @classmethod
    def setUpTestData(cls):
        if False:
            i = 10
            return i + 15
        device = create_test_device('device1')
        interfaces = (Interface(device=device, name='eth0'), Interface(device=device, name='eth1'), Interface(device=device, name='eth2'))
        Interface.objects.bulk_create(interfaces)
        virtual_machine = create_test_virtualmachine('virtual_machine1')
        vm_interfaces = (VMInterface(virtual_machine=virtual_machine, name='eth0'), VMInterface(virtual_machine=virtual_machine, name='eth1'), VMInterface(virtual_machine=virtual_machine, name='eth2'))
        VMInterface.objects.bulk_create(vm_interfaces)
        fhrp_groups = (FHRPGroup(protocol=FHRPGroupProtocolChoices.PROTOCOL_VRRP2, group_id=10), FHRPGroup(protocol=FHRPGroupProtocolChoices.PROTOCOL_VRRP3, group_id=20), FHRPGroup(protocol=FHRPGroupProtocolChoices.PROTOCOL_HSRP, group_id=30))
        FHRPGroup.objects.bulk_create(fhrp_groups)
        fhrp_group_assignments = (FHRPGroupAssignment(group=fhrp_groups[0], interface=interfaces[0], priority=10), FHRPGroupAssignment(group=fhrp_groups[1], interface=interfaces[1], priority=20), FHRPGroupAssignment(group=fhrp_groups[2], interface=interfaces[2], priority=30), FHRPGroupAssignment(group=fhrp_groups[0], interface=vm_interfaces[0], priority=10), FHRPGroupAssignment(group=fhrp_groups[1], interface=vm_interfaces[1], priority=20), FHRPGroupAssignment(group=fhrp_groups[2], interface=vm_interfaces[2], priority=30))
        FHRPGroupAssignment.objects.bulk_create(fhrp_group_assignments)

    def test_group_id(self):
        if False:
            return 10
        fhrp_groups = FHRPGroup.objects.all()[:2]
        params = {'group_id': [fhrp_groups[0].pk, fhrp_groups[1].pk]}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 4)

    def test_interface_type(self):
        if False:
            return 10
        params = {'interface_type': 'dcim.interface'}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 3)

    def test_interface(self):
        if False:
            for i in range(10):
                print('nop')
        interfaces = Interface.objects.all()[:2]
        params = {'interface_type': 'dcim.interface', 'interface_id': [interfaces[0].pk, interfaces[1].pk]}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 2)

    def test_priority(self):
        if False:
            for i in range(10):
                print('nop')
        params = {'priority': [10, 20]}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 4)

    def test_device(self):
        if False:
            return 10
        device = Device.objects.first()
        params = {'device': [device.name]}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 3)
        params = {'device_id': [device.pk]}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 3)

    def test_virtual_machine(self):
        if False:
            i = 10
            return i + 15
        vm = VirtualMachine.objects.first()
        params = {'virtual_machine': [vm.name]}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 3)
        params = {'virtual_machine_id': [vm.pk]}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 3)

class VLANGroupTestCase(TestCase, ChangeLoggedFilterSetTests):
    queryset = VLANGroup.objects.all()
    filterset = VLANGroupFilterSet

    @classmethod
    def setUpTestData(cls):
        if False:
            for i in range(10):
                print('nop')
        region = Region(name='Region 1', slug='region-1')
        region.save()
        sitegroup = SiteGroup(name='Site Group 1', slug='site-group-1')
        sitegroup.save()
        site = Site(name='Site 1', slug='site-1')
        site.save()
        location = Location(name='Location 1', slug='location-1', site=site)
        location.save()
        rack = Rack(name='Rack 1', site=site)
        rack.save()
        clustertype = ClusterType(name='Cluster Type 1', slug='cluster-type-1')
        clustertype.save()
        clustergroup = ClusterGroup(name='Cluster Group 1', slug='cluster-group-1')
        clustergroup.save()
        cluster = Cluster(name='Cluster 1', type=clustertype)
        cluster.save()
        vlan_groups = (VLANGroup(name='VLAN Group 1', slug='vlan-group-1', scope=region, description='A'), VLANGroup(name='VLAN Group 2', slug='vlan-group-2', scope=sitegroup, description='B'), VLANGroup(name='VLAN Group 3', slug='vlan-group-3', scope=site, description='C'), VLANGroup(name='VLAN Group 4', slug='vlan-group-4', scope=location, description='D'), VLANGroup(name='VLAN Group 5', slug='vlan-group-5', scope=rack, description='E'), VLANGroup(name='VLAN Group 6', slug='vlan-group-6', scope=clustergroup, description='F'), VLANGroup(name='VLAN Group 7', slug='vlan-group-7', scope=cluster, description='G'), VLANGroup(name='VLAN Group 8', slug='vlan-group-8'))
        VLANGroup.objects.bulk_create(vlan_groups)

    def test_name(self):
        if False:
            for i in range(10):
                print('nop')
        params = {'name': ['VLAN Group 1', 'VLAN Group 2']}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 2)

    def test_slug(self):
        if False:
            return 10
        params = {'slug': ['vlan-group-1', 'vlan-group-2']}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 2)

    def test_description(self):
        if False:
            print('Hello World!')
        params = {'description': ['A', 'B']}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 2)

    def test_region(self):
        if False:
            return 10
        params = {'region': Region.objects.first().pk}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 1)

    def test_sitegroup(self):
        if False:
            while True:
                i = 10
        params = {'sitegroup': SiteGroup.objects.first().pk}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 1)

    def test_site(self):
        if False:
            while True:
                i = 10
        params = {'site': Site.objects.first().pk}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 1)

    def test_location(self):
        if False:
            for i in range(10):
                print('nop')
        params = {'location': Location.objects.first().pk}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 1)

    def test_rack(self):
        if False:
            while True:
                i = 10
        params = {'rack': Rack.objects.first().pk}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 1)

    def test_clustergroup(self):
        if False:
            i = 10
            return i + 15
        params = {'clustergroup': ClusterGroup.objects.first().pk}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 1)

    def test_cluster(self):
        if False:
            i = 10
            return i + 15
        params = {'cluster': Cluster.objects.first().pk}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 1)

class VLANTestCase(TestCase, ChangeLoggedFilterSetTests):
    queryset = VLAN.objects.all()
    filterset = VLANFilterSet

    @classmethod
    def setUpTestData(cls):
        if False:
            print('Hello World!')
        regions = (Region(name='Test Region 1', slug='test-region-1'), Region(name='Test Region 2', slug='test-region-2'), Region(name='Test Region 3', slug='test-region-3'))
        for r in regions:
            r.save()
        site_groups = (SiteGroup(name='Site Group 1', slug='site-group-1'), SiteGroup(name='Site Group 2', slug='site-group-2'), SiteGroup(name='Site Group 3', slug='site-group-3'))
        for site_group in site_groups:
            site_group.save()
        sites = (Site(name='Site 1', slug='site-1', region=regions[0], group=site_groups[0]), Site(name='Site 2', slug='site-2', region=regions[1], group=site_groups[1]), Site(name='Site 3', slug='site-3', region=regions[2], group=site_groups[2]), Site(name='Site 4', slug='site-4', region=regions[0], group=site_groups[0]), Site(name='Site 5', slug='site-5', region=regions[1], group=site_groups[1]), Site(name='Site 6', slug='site-6', region=regions[2], group=site_groups[2]))
        Site.objects.bulk_create(sites)
        locations = (Location(name='Location 1', slug='location-1', site=sites[0]), Location(name='Location 2', slug='location-2', site=sites[1]), Location(name='Location 3', slug='location-3', site=sites[2]))
        for location in locations:
            location.save()
        racks = (Rack(name='Rack 1', site=sites[0], location=locations[0]), Rack(name='Rack 2', site=sites[1], location=locations[1]), Rack(name='Rack 3', site=sites[2], location=locations[2]))
        Rack.objects.bulk_create(racks)
        manufacturer = Manufacturer.objects.create(name='Manufacturer 1', slug='manufacturer-1')
        device_type = DeviceType.objects.create(manufacturer=manufacturer, model='Device Type 1')
        role = DeviceRole.objects.create(name='Device Role 1', slug='device-role-1')
        devices = (Device(name='Device 1', site=sites[0], location=locations[0], rack=racks[0], device_type=device_type, role=role), Device(name='Device 2', site=sites[1], location=locations[1], rack=racks[1], device_type=device_type, role=role), Device(name='Device 3', site=sites[2], location=locations[2], rack=racks[2], device_type=device_type, role=role))
        Device.objects.bulk_create(devices)
        cluster_groups = (ClusterGroup(name='Cluster Group 1', slug='cluster-group-1'), ClusterGroup(name='Cluster Group 2', slug='cluster-group-2'), ClusterGroup(name='Cluster Group 3', slug='cluster-group-3'))
        ClusterGroup.objects.bulk_create(cluster_groups)
        cluster_type = ClusterType.objects.create(name='Cluster Type 1', slug='cluster-type-1')
        clusters = (Cluster(name='Cluster 1', type=cluster_type, group=cluster_groups[0], site=sites[0]), Cluster(name='Cluster 2', type=cluster_type, group=cluster_groups[1], site=sites[1]), Cluster(name='Cluster 3', type=cluster_type, group=cluster_groups[2], site=sites[2]))
        Cluster.objects.bulk_create(clusters)
        virtual_machines = (VirtualMachine(name='Virtual Machine 1', cluster=clusters[0]), VirtualMachine(name='Virtual Machine 2', cluster=clusters[1]), VirtualMachine(name='Virtual Machine 3', cluster=clusters[2]))
        VirtualMachine.objects.bulk_create(virtual_machines)
        groups = (VLANGroup(name='Region 1', slug='region-1', scope=regions[0]), VLANGroup(name='Region 2', slug='region-2', scope=regions[1]), VLANGroup(name='Region 3', slug='region-3', scope=regions[2]), VLANGroup(name='Site Group 1', slug='site-group-1', scope=site_groups[0]), VLANGroup(name='Site Group 2', slug='site-group-2', scope=site_groups[1]), VLANGroup(name='Site Group 3', slug='site-group-3', scope=site_groups[2]), VLANGroup(name='Site 1', slug='site-1', scope=sites[0]), VLANGroup(name='Site 2', slug='site-2', scope=sites[1]), VLANGroup(name='Site 3', slug='site-3', scope=sites[2]), VLANGroup(name='Location 1', slug='location-1', scope=locations[0]), VLANGroup(name='Location 2', slug='location-2', scope=locations[1]), VLANGroup(name='Location 3', slug='location-3', scope=locations[2]), VLANGroup(name='Rack 1', slug='rack-1', scope=racks[0]), VLANGroup(name='Rack 2', slug='rack-2', scope=racks[1]), VLANGroup(name='Rack 3', slug='rack-3', scope=racks[2]), VLANGroup(name='Cluster Group 1', slug='cluster-group-1', scope=cluster_groups[0]), VLANGroup(name='Cluster Group 2', slug='cluster-group-2', scope=cluster_groups[1]), VLANGroup(name='Cluster Group 3', slug='cluster-group-3', scope=cluster_groups[2]), VLANGroup(name='Cluster 1', slug='cluster-1', scope=clusters[0]), VLANGroup(name='Cluster 2', slug='cluster-2', scope=clusters[1]), VLANGroup(name='Cluster 3', slug='cluster-3', scope=clusters[2]), VLANGroup(name='VLAN Group 1', slug='vlan-group-1'), VLANGroup(name='VLAN Group 2', slug='vlan-group-2'), VLANGroup(name='VLAN Group 3', slug='vlan-group-3'))
        VLANGroup.objects.bulk_create(groups)
        roles = (Role(name='Role 1', slug='role-1'), Role(name='Role 2', slug='role-2'), Role(name='Role 3', slug='role-3'))
        Role.objects.bulk_create(roles)
        tenant_groups = (TenantGroup(name='Tenant group 1', slug='tenant-group-1'), TenantGroup(name='Tenant group 2', slug='tenant-group-2'), TenantGroup(name='Tenant group 3', slug='tenant-group-3'))
        for tenantgroup in tenant_groups:
            tenantgroup.save()
        tenants = (Tenant(name='Tenant 1', slug='tenant-1', group=tenant_groups[0]), Tenant(name='Tenant 2', slug='tenant-2', group=tenant_groups[1]), Tenant(name='Tenant 3', slug='tenant-3', group=tenant_groups[2]))
        Tenant.objects.bulk_create(tenants)
        vlans = (VLAN(vid=1, name='Region 1', group=groups[0], description='foobar1'), VLAN(vid=2, name='Region 2', group=groups[1], description='foobar2'), VLAN(vid=3, name='Region 3', group=groups[2]), VLAN(vid=4, name='Site Group 1', group=groups[3]), VLAN(vid=5, name='Site Group 2', group=groups[4]), VLAN(vid=6, name='Site Group 3', group=groups[5]), VLAN(vid=7, name='Site 1', group=groups[6]), VLAN(vid=8, name='Site 2', group=groups[7]), VLAN(vid=9, name='Site 3', group=groups[8]), VLAN(vid=10, name='Location 1', group=groups[9]), VLAN(vid=11, name='Location 2', group=groups[10]), VLAN(vid=12, name='Location 3', group=groups[11]), VLAN(vid=13, name='Rack 1', group=groups[12]), VLAN(vid=14, name='Rack 2', group=groups[13]), VLAN(vid=15, name='Rack 3', group=groups[14]), VLAN(vid=16, name='Cluster Group 1', group=groups[15]), VLAN(vid=17, name='Cluster Group 2', group=groups[16]), VLAN(vid=18, name='Cluster Group 3', group=groups[17]), VLAN(vid=19, name='Cluster 1', group=groups[18]), VLAN(vid=20, name='Cluster 2', group=groups[19]), VLAN(vid=21, name='Cluster 3', group=groups[20]), VLAN(vid=101, name='VLAN 101', site=sites[3], group=groups[21], role=roles[0], tenant=tenants[0], status=VLANStatusChoices.STATUS_ACTIVE), VLAN(vid=102, name='VLAN 102', site=sites[3], group=groups[21], role=roles[0], tenant=tenants[0], status=VLANStatusChoices.STATUS_ACTIVE), VLAN(vid=201, name='VLAN 201', site=sites[4], group=groups[22], role=roles[1], tenant=tenants[1], status=VLANStatusChoices.STATUS_DEPRECATED), VLAN(vid=202, name='VLAN 202', site=sites[4], group=groups[22], role=roles[1], tenant=tenants[1], status=VLANStatusChoices.STATUS_DEPRECATED), VLAN(vid=301, name='VLAN 301', site=sites[5], group=groups[23], role=roles[2], tenant=tenants[2], status=VLANStatusChoices.STATUS_RESERVED), VLAN(vid=302, name='VLAN 302', site=sites[5], group=groups[23], role=roles[2], tenant=tenants[2], status=VLANStatusChoices.STATUS_RESERVED), VLAN(vid=1000, name='Global VLAN'))
        VLAN.objects.bulk_create(vlans)

    def test_name(self):
        if False:
            while True:
                i = 10
        params = {'name': ['VLAN 101', 'VLAN 102']}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 2)

    def test_vid(self):
        if False:
            return 10
        params = {'vid': ['101', '201', '301']}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 3)

    def test_region(self):
        if False:
            for i in range(10):
                print('nop')
        regions = Region.objects.all()[:2]
        params = {'region_id': [regions[0].pk, regions[1].pk]}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 4)
        params = {'region': [regions[0].slug, regions[1].slug]}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 4)

    def test_site_group(self):
        if False:
            print('Hello World!')
        site_groups = SiteGroup.objects.all()[:2]
        params = {'site_group_id': [site_groups[0].pk, site_groups[1].pk]}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 4)
        params = {'site_group': [site_groups[0].slug, site_groups[1].slug]}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 4)

    def test_site(self):
        if False:
            while True:
                i = 10
        sites = Site.objects.all()
        params = {'site_id': [sites[3].pk, sites[4].pk]}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 4)
        params = {'site': [sites[3].slug, sites[4].slug]}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 4)

    def test_group(self):
        if False:
            i = 10
            return i + 15
        groups = VLANGroup.objects.filter(name__startswith='VLAN Group')[:2]
        params = {'group_id': [groups[0].pk, groups[1].pk]}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 4)
        params = {'group': [groups[0].slug, groups[1].slug]}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 4)

    def test_description(self):
        if False:
            while True:
                i = 10
        params = {'description': ['foobar1', 'foobar2']}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 2)

    def test_role(self):
        if False:
            print('Hello World!')
        roles = Role.objects.all()[:2]
        params = {'role_id': [roles[0].pk, roles[1].pk]}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 4)
        params = {'role': [roles[0].slug, roles[1].slug]}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 4)

    def test_status(self):
        if False:
            while True:
                i = 10
        params = {'status': [VLANStatusChoices.STATUS_DEPRECATED, VLANStatusChoices.STATUS_RESERVED]}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 4)

    def test_tenant(self):
        if False:
            i = 10
            return i + 15
        tenants = Tenant.objects.all()[:2]
        params = {'tenant_id': [tenants[0].pk, tenants[1].pk]}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 4)
        params = {'tenant': [tenants[0].slug, tenants[1].slug]}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 4)

    def test_tenant_group(self):
        if False:
            i = 10
            return i + 15
        tenant_groups = TenantGroup.objects.all()[:2]
        params = {'tenant_group_id': [tenant_groups[0].pk, tenant_groups[1].pk]}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 4)
        params = {'tenant_group': [tenant_groups[0].slug, tenant_groups[1].slug]}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 4)

    def test_available_on_device(self):
        if False:
            return 10
        device_id = Device.objects.first().pk
        params = {'available_on_device': device_id}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 6)

    def test_available_on_virtualmachine(self):
        if False:
            return 10
        vm_id = VirtualMachine.objects.first().pk
        params = {'available_on_virtualmachine': vm_id}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 6)

class ServiceTemplateTestCase(TestCase, ChangeLoggedFilterSetTests):
    queryset = ServiceTemplate.objects.all()
    filterset = ServiceTemplateFilterSet

    @classmethod
    def setUpTestData(cls):
        if False:
            i = 10
            return i + 15
        service_templates = (ServiceTemplate(name='Service Template 1', protocol=ServiceProtocolChoices.PROTOCOL_TCP, ports=[1001]), ServiceTemplate(name='Service Template 2', protocol=ServiceProtocolChoices.PROTOCOL_TCP, ports=[1002]), ServiceTemplate(name='Service Template 3', protocol=ServiceProtocolChoices.PROTOCOL_UDP, ports=[1003]), ServiceTemplate(name='Service Template 4', protocol=ServiceProtocolChoices.PROTOCOL_TCP, ports=[2001]), ServiceTemplate(name='Service Template 5', protocol=ServiceProtocolChoices.PROTOCOL_TCP, ports=[2002]), ServiceTemplate(name='Service Template 6', protocol=ServiceProtocolChoices.PROTOCOL_UDP, ports=[2003]))
        ServiceTemplate.objects.bulk_create(service_templates)

    def test_name(self):
        if False:
            i = 10
            return i + 15
        params = {'name': ['Service Template 1', 'Service Template 2']}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 2)

    def test_protocol(self):
        if False:
            while True:
                i = 10
        params = {'protocol': ServiceProtocolChoices.PROTOCOL_TCP}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 4)

    def test_port(self):
        if False:
            while True:
                i = 10
        params = {'port': '1001'}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 1)

class ServiceTestCase(TestCase, ChangeLoggedFilterSetTests):
    queryset = Service.objects.all()
    filterset = ServiceFilterSet

    @classmethod
    def setUpTestData(cls):
        if False:
            i = 10
            return i + 15
        site = Site.objects.create(name='Site 1', slug='site-1')
        manufacturer = Manufacturer.objects.create(name='Manufacturer 1', slug='manufacturer-1')
        device_type = DeviceType.objects.create(manufacturer=manufacturer, model='Device Type 1')
        role = DeviceRole.objects.create(name='Device Role 1', slug='device-role-1')
        devices = (Device(device_type=device_type, name='Device 1', site=site, role=role), Device(device_type=device_type, name='Device 2', site=site, role=role), Device(device_type=device_type, name='Device 3', site=site, role=role))
        Device.objects.bulk_create(devices)
        interface = Interface.objects.create(device=devices[0], name='eth0', type=InterfaceTypeChoices.TYPE_VIRTUAL)
        interface_ct = ContentType.objects.get_for_model(Interface).pk
        ip_addresses = (IPAddress(address='192.0.2.1/24', assigned_object_type_id=interface_ct, assigned_object_id=interface.pk), IPAddress(address='192.0.2.2/24', assigned_object_type_id=interface_ct, assigned_object_id=interface.pk), IPAddress(address='192.0.2.3/24', assigned_object_type_id=interface_ct, assigned_object_id=interface.pk))
        IPAddress.objects.bulk_create(ip_addresses)
        clustertype = ClusterType.objects.create(name='Cluster Type 1', slug='cluster-type-1')
        cluster = Cluster.objects.create(type=clustertype, name='Cluster 1')
        virtual_machines = (VirtualMachine(name='Virtual Machine 1', cluster=cluster), VirtualMachine(name='Virtual Machine 2', cluster=cluster), VirtualMachine(name='Virtual Machine 3', cluster=cluster))
        VirtualMachine.objects.bulk_create(virtual_machines)
        services = (Service(device=devices[0], name='Service 1', protocol=ServiceProtocolChoices.PROTOCOL_TCP, ports=[1001], description='foobar1'), Service(device=devices[1], name='Service 2', protocol=ServiceProtocolChoices.PROTOCOL_TCP, ports=[1002], description='foobar2'), Service(device=devices[2], name='Service 3', protocol=ServiceProtocolChoices.PROTOCOL_UDP, ports=[1003]), Service(virtual_machine=virtual_machines[0], name='Service 4', protocol=ServiceProtocolChoices.PROTOCOL_TCP, ports=[2001]), Service(virtual_machine=virtual_machines[1], name='Service 5', protocol=ServiceProtocolChoices.PROTOCOL_TCP, ports=[2002]), Service(virtual_machine=virtual_machines[2], name='Service 6', protocol=ServiceProtocolChoices.PROTOCOL_UDP, ports=[2003]))
        Service.objects.bulk_create(services)
        services[0].ipaddresses.add(ip_addresses[0])
        services[1].ipaddresses.add(ip_addresses[1])
        services[2].ipaddresses.add(ip_addresses[2])

    def test_name(self):
        if False:
            i = 10
            return i + 15
        params = {'name': ['Service 1', 'Service 2']}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 2)

    def test_protocol(self):
        if False:
            return 10
        params = {'protocol': ServiceProtocolChoices.PROTOCOL_TCP}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 4)

    def test_description(self):
        if False:
            print('Hello World!')
        params = {'description': ['foobar1', 'foobar2']}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 2)

    def test_port(self):
        if False:
            print('Hello World!')
        params = {'port': '1001'}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 1)

    def test_device(self):
        if False:
            while True:
                i = 10
        devices = Device.objects.all()[:2]
        params = {'device_id': [devices[0].pk, devices[1].pk]}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 2)
        params = {'device': [devices[0].name, devices[1].name]}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 2)

    def test_virtual_machine(self):
        if False:
            print('Hello World!')
        vms = VirtualMachine.objects.all()[:2]
        params = {'virtual_machine_id': [vms[0].pk, vms[1].pk]}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 2)
        params = {'virtual_machine': [vms[0].name, vms[1].name]}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 2)

    def test_ipaddress(self):
        if False:
            return 10
        ips = IPAddress.objects.all()[:2]
        params = {'ipaddress_id': [ips[0].pk, ips[1].pk]}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 2)
        params = {'ipaddress': [str(ips[0].address), str(ips[1].address)]}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 2)

class L2VPNTestCase(TestCase, ChangeLoggedFilterSetTests):
    queryset = L2VPN.objects.all()
    filterset = L2VPNFilterSet

    @classmethod
    def setUpTestData(cls):
        if False:
            print('Hello World!')
        route_targets = (RouteTarget(name='1:1'), RouteTarget(name='1:2'), RouteTarget(name='1:3'), RouteTarget(name='2:1'), RouteTarget(name='2:2'), RouteTarget(name='2:3'))
        RouteTarget.objects.bulk_create(route_targets)
        l2vpns = (L2VPN(name='L2VPN 1', slug='l2vpn-1', type=L2VPNTypeChoices.TYPE_VXLAN, identifier=65001), L2VPN(name='L2VPN 2', slug='l2vpn-2', type=L2VPNTypeChoices.TYPE_VPWS, identifier=65002), L2VPN(name='L2VPN 3', slug='l2vpn-3', type=L2VPNTypeChoices.TYPE_VPLS))
        L2VPN.objects.bulk_create(l2vpns)
        l2vpns[0].import_targets.add(route_targets[0])
        l2vpns[1].import_targets.add(route_targets[1])
        l2vpns[2].import_targets.add(route_targets[2])
        l2vpns[0].export_targets.add(route_targets[3])
        l2vpns[1].export_targets.add(route_targets[4])
        l2vpns[2].export_targets.add(route_targets[5])

    def test_name(self):
        if False:
            for i in range(10):
                print('nop')
        params = {'name': ['L2VPN 1', 'L2VPN 2']}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 2)

    def test_slug(self):
        if False:
            return 10
        params = {'slug': ['l2vpn-1', 'l2vpn-2']}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 2)

    def test_identifier(self):
        if False:
            i = 10
            return i + 15
        params = {'identifier': ['65001', '65002']}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 2)

    def test_type(self):
        if False:
            print('Hello World!')
        params = {'type': [L2VPNTypeChoices.TYPE_VXLAN, L2VPNTypeChoices.TYPE_VPWS]}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 2)

    def test_import_targets(self):
        if False:
            i = 10
            return i + 15
        route_targets = RouteTarget.objects.filter(name__in=['1:1', '1:2'])
        params = {'import_target_id': [route_targets[0].pk, route_targets[1].pk]}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 2)
        params = {'import_target': [route_targets[0].name, route_targets[1].name]}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 2)

    def test_export_targets(self):
        if False:
            for i in range(10):
                print('nop')
        route_targets = RouteTarget.objects.filter(name__in=['2:1', '2:2'])
        params = {'export_target_id': [route_targets[0].pk, route_targets[1].pk]}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 2)
        params = {'export_target': [route_targets[0].name, route_targets[1].name]}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 2)

class L2VPNTerminationTestCase(TestCase, ChangeLoggedFilterSetTests):
    queryset = L2VPNTermination.objects.all()
    filterset = L2VPNTerminationFilterSet

    @classmethod
    def setUpTestData(cls):
        if False:
            return 10
        device = create_test_device('Device 1')
        interfaces = (Interface(name='Interface 1', device=device, type=InterfaceTypeChoices.TYPE_1GE_FIXED), Interface(name='Interface 2', device=device, type=InterfaceTypeChoices.TYPE_1GE_FIXED), Interface(name='Interface 3', device=device, type=InterfaceTypeChoices.TYPE_1GE_FIXED))
        Interface.objects.bulk_create(interfaces)
        vm = create_test_virtualmachine('Virtual Machine 1')
        vminterfaces = (VMInterface(name='Interface 1', virtual_machine=vm), VMInterface(name='Interface 2', virtual_machine=vm), VMInterface(name='Interface 3', virtual_machine=vm))
        VMInterface.objects.bulk_create(vminterfaces)
        vlans = (VLAN(name='VLAN 1', vid=101), VLAN(name='VLAN 2', vid=102), VLAN(name='VLAN 3', vid=103))
        VLAN.objects.bulk_create(vlans)
        l2vpns = (L2VPN(name='L2VPN 1', slug='l2vpn-1', type='vxlan', identifier=65001), L2VPN(name='L2VPN 2', slug='l2vpn-2', type='vpws', identifier=65002), L2VPN(name='L2VPN 3', slug='l2vpn-3', type='vpls'))
        L2VPN.objects.bulk_create(l2vpns)
        l2vpnterminations = (L2VPNTermination(l2vpn=l2vpns[0], assigned_object=vlans[0]), L2VPNTermination(l2vpn=l2vpns[1], assigned_object=vlans[1]), L2VPNTermination(l2vpn=l2vpns[2], assigned_object=vlans[2]), L2VPNTermination(l2vpn=l2vpns[0], assigned_object=interfaces[0]), L2VPNTermination(l2vpn=l2vpns[1], assigned_object=interfaces[1]), L2VPNTermination(l2vpn=l2vpns[2], assigned_object=interfaces[2]), L2VPNTermination(l2vpn=l2vpns[0], assigned_object=vminterfaces[0]), L2VPNTermination(l2vpn=l2vpns[1], assigned_object=vminterfaces[1]), L2VPNTermination(l2vpn=l2vpns[2], assigned_object=vminterfaces[2]))
        L2VPNTermination.objects.bulk_create(l2vpnterminations)

    def test_l2vpn(self):
        if False:
            i = 10
            return i + 15
        l2vpns = L2VPN.objects.all()[:2]
        params = {'l2vpn_id': [l2vpns[0].pk, l2vpns[1].pk]}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 6)
        params = {'l2vpn': [l2vpns[0].slug, l2vpns[1].slug]}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 6)

    def test_content_type(self):
        if False:
            for i in range(10):
                print('nop')
        params = {'assigned_object_type_id': ContentType.objects.get(model='vlan').pk}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 3)

    def test_interface(self):
        if False:
            return 10
        interfaces = Interface.objects.all()[:2]
        params = {'interface_id': [interfaces[0].pk, interfaces[1].pk]}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 2)

    def test_vminterface(self):
        if False:
            print('Hello World!')
        vminterfaces = VMInterface.objects.all()[:2]
        params = {'vminterface_id': [vminterfaces[0].pk, vminterfaces[1].pk]}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 2)

    def test_vlan(self):
        if False:
            while True:
                i = 10
        vlans = VLAN.objects.all()[:2]
        params = {'vlan_id': [vlans[0].pk, vlans[1].pk]}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 2)
        params = {'vlan': ['VLAN 1', 'VLAN 2']}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 2)

    def test_site(self):
        if False:
            return 10
        site = Site.objects.all().first()
        params = {'site_id': [site.pk]}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 3)
        params = {'site': ['site-1']}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 3)

    def test_device(self):
        if False:
            return 10
        device = Device.objects.all().first()
        params = {'device_id': [device.pk]}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 3)
        params = {'device': ['Device 1']}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 3)

    def test_virtual_machine(self):
        if False:
            return 10
        virtual_machine = VirtualMachine.objects.all().first()
        params = {'virtual_machine_id': [virtual_machine.pk]}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 3)
        params = {'virtual_machine': ['Virtual Machine 1']}
        self.assertEqual(self.filterset(params, self.queryset).qs.count(), 3)